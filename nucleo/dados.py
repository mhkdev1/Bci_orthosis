# -*- coding: utf-8 -*-
"""
dados.py — leitura das duas origens, com a MESMA geometria de janela.

Origens:
  - EDF do PhysioNet (eegmmidb), em `dados/physionet/Sxxx/`
  - gravação contínua da Cyton, em `dados/gravacoes/<sessão>/`

As duas devolvem janelas de 4,1 s centradas nas transições, no formato
(n, 3 canais, 526 amostras). Quem chama não precisa saber de onde veio.
"""
from __future__ import annotations

import glob
import os
import re

import numpy as np

import config as cfg
from nucleo import preproc


# =====================================================================
# PhysioNet (EDF)
# =====================================================================
_RE_EDF = re.compile(r"S(\d{3})R(\d{2})\.edf$", re.I)


def sujeitos_physionet(dir_base=None):
    """{'S001': {3: caminho, 4: caminho, ...}, ...} do que existir em disco."""
    idx = {}
    for p in glob.glob(os.path.join(dir_base or cfg.DIR_PHYSIONET, "**", "*.edf"), recursive=True):
        m = _RE_EDF.search(os.path.basename(p))
        if m:
            idx.setdefault(f"S{m[1]}", {})[int(m[2])] = p
    return {s: dict(sorted(r.items())) for s, r in sorted(idx.items())}


def janelas_edf(caminho):
    """Um arquivo EDF -> (X, y, t). Janelas centradas nas transições.

    `t` é o instante (em segundos, dentro do run) do centro de cada janela — ou seja, o
    momento da transição. Serve para localizar um trial específico na gravação.

    A janela do movimento é centrada no início da tarefa (anotações T1/T2) e a do
    descanso no início do descanso (T0). Como o protocolo alterna descanso e tarefa com
    a mesma duração, o início de um segmento é sempre uma transição.

    baseline=None nas duas classes, de propósito: corrigir baseline só numa delas criaria
    uma diferença de pré-processamento correlacionada com o rótulo, e a rede aprenderia
    isso em vez do sinal. O detrend linear já cobre offset e deriva.
    """
    import gc
    import warnings

    import mne
    mne.set_log_level("ERROR")
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    raw = mne.io.read_raw_edf(caminho, preload=False, verbose=False)
    mne.datasets.eegbci.standardize(raw)
    raw.set_montage("standard_1005", on_missing="ignore")

    nomes = {c.lower(): c for c in raw.ch_names}
    sel = [nomes[n.lower()] for n in cfg.CANAIS if n.lower() in nomes]
    if len(sel) != len(cfg.CANAIS):
        return None
    raw.pick_channels(sel, ordered=True)
    raw.load_data()
    _limpar_anotacoes(raw)
    preproc.cadeia_mne(raw)

    eventos, _ = mne.events_from_annotations(
        raw, event_id={"T0": 3, "T1": 1, "T2": 2}, verbose=False)
    if len(eventos) == 0:
        return None

    meia = cfg.MEIA_JANELA
    taxa = float(raw.info["sfreq"])
    Xs, ys, ts = [], [], []
    for codigos, mapa in (((1, 2), {1: 1, 2: 2}), ((3,), {3: 0})):
        ev = eventos[np.isin(eventos[:, 2], codigos)]
        if len(ev) == 0:
            continue
        ep = mne.Epochs(raw, ev, event_id=None, tmin=-meia, tmax=+meia,
                        baseline=None, preload=True, detrend=1,
                        reject_by_annotation=True, verbose=False)
        if len(ep) == 0:
            continue
        # O mne.Epochs devolve round((tmax-tmin)*taxa)+1 amostras, que com 4,1 s a 128 Hz
        # dá 525 — uma a menos que as 526 do config, porque 4,1 x 128 = 524,8 não é
        # inteiro. A diferença de uma amostra (0,2% da janela) é completada aqui, para
        # que EDF e gravação cheguem à rede com exatamente o mesmo comprimento.
        Xs.append(preproc.ajustar_lote(ep.get_data().astype(np.float32)))
        ys.append(np.array([mapa[int(c)] for c in ep.events[:, 2]], dtype=np.int64))
        ts.append(ep.events[:, 0].astype(np.float64) / taxa)

    del raw
    gc.collect()
    if not Xs:
        return None
    X, y, t = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ts)
    ordem = np.argsort(t, kind="stable")      # devolve na ordem em que acontecem no run
    return X[ordem], y[ordem], t[ordem]


def _limpar_anotacoes(raw):
    """Remove anotações que caem fora do intervalo gravado."""
    fora = [i for i, (o, d) in enumerate(zip(raw.annotations.onset, raw.annotations.duration))
            if (o < 0 and o + d <= 0) or o >= raw.times[-1]]
    if fora:
        raw.set_annotations(raw.annotations[[i for i in range(len(raw.annotations))
                                             if i not in fora]])


# =====================================================================
# Gravações da Cyton (contínuas)
# =====================================================================
def sessoes_gravadas(dir_base=None):
    """['S004_R1', 'S005_R1_cortado', ...] — pastas com continuo.csv + eventos.csv."""
    base = dir_base or cfg.DIR_GRAVACOES
    return sorted(os.path.basename(os.path.dirname(p))
                  for p in glob.glob(os.path.join(base, "*", "eventos.csv")))


def janelas_gravacao(pasta, arquivo="continuo.csv"):
    """Uma sessão gravada -> (X, y, info). Mesma geometria do EDF.

    Cada janela é centrada no início de um segmento, o que exige metade do segmento
    ANTERIOR. O primeiro evento da sessão não tem isso e é descartado — daí a contagem
    de janelas ser sempre uma a menos que a de eventos.
    """
    import pandas as pd
    ev = pd.read_csv(os.path.join(pasta, "eventos.csv"))
    sinal = pd.read_csv(os.path.join(pasta, arquivo))[
        [f"{c}_uV" for c in cfg.CANAIS]].to_numpy(dtype=np.float64)

    taxa = float(cfg.TAXA_PLACA)
    largura = int(np.ceil(cfg.AMOSTRAS / cfg.TAXA_IA * taxa))
    meia = largura // 2

    Xs, ys, info = [], [], []
    for _, r in ev.iterrows():
        classe = cfg.CODIGO_CLASSE.get(str(r["code"]))
        if classe is None:
            continue
        ini = int(r["amostra_inicio"]) - meia
        if ini < 0 or ini + largura > len(sinal):
            info.append({"trial": int(r["trial_index"]), "code": str(r["code"]),
                         "usada": False, "t_s": float(r["amostra_inicio"]) / taxa})
            continue
        Xs.append(preproc.janela_bruta(sinal[ini:ini + largura].T * 1e-6, taxa)[None])
        ys.append(classe)
        info.append({"trial": int(r["trial_index"]), "code": str(r["code"]),
                     "usada": True, "t_s": float(r["amostra_inicio"]) / taxa})
    if not Xs:
        return None
    return (np.concatenate(Xs).astype(np.float32),
            np.array(ys, dtype=np.int64), info)


def sinal_continuo(pasta, arquivo="continuo.csv"):
    """(sinal (L,C) em Volts, eventos DataFrame) — para a varredura por janela deslizante."""
    import pandas as pd
    ev = pd.read_csv(os.path.join(pasta, "eventos.csv"))
    d = pd.read_csv(os.path.join(pasta, arquivo))
    x = d[[f"{c}_uV" for c in cfg.CANAIS]].to_numpy(dtype=np.float64) * 1e-6
    return x, ev


def bons_sujeitos(sujeitos):
    """Tira os sujeitos defeituosos do eegmmidb (gravados a 128 Hz ou run corrompido)."""
    ruins = {f"S{n:03d}" for n in cfg.SUJEITOS_RUINS}
    return [s for s in sujeitos if s not in ruins]
