# -*- coding: utf-8 -*-
"""
preproc.py — a cadeia de pré-processamento, idêntica para todas as origens de dado.

Esta é a regra que sustenta o projeto inteiro: **EDF do PhysioNet e gravação da Cyton
passam exatamente pelo mesmo caminho**. Se o tratamento divergir entre os dois, o modelo
aprende a diferença entre datasets em vez de aprender a diferença entre classes, e o
número de validação deixa de dizer qualquer coisa sobre o uso real.

A cadeia:

    notch 60 Hz -> passa-banda 0,5-35 Hz (FIR firwin) -> reamostra para 128 Hz
    -> recorta a janela -> detrend linear -> z-score por canal

O z-score usa a média e o desvio calculados **no conjunto de treino** e gravados no
checkpoint. Na inferência, aplica-se essa mesma estatística. É o que garante que treino
e uso vejam a mesma escala.
"""
from __future__ import annotations

import numpy as np

import config as cfg


def cadeia_mne(raw):
    """notch -> passa-banda -> reamostragem, sobre um objeto Raw do MNE.

    Sem referência média (CAR): com apenas 3 eletrodos a CAR não aproxima uma referência
    neutra — ela força C3+C4+Cz=0 e derruba o posto de 3 para 2, entregando à rede três
    canais com só duas dimensões de informação (medido: o terceiro valor singular cai
    para ~1e-12 contra ~4e-4 dos outros). Fazer CAR direito exigiria a cabeça inteira,
    que a montagem de 3 eletrodos não tem.
    """
    raw.notch_filter(cfg.NOTCH, picks="all", verbose=False)
    raw.filter(cfg.PASSA_BANDA[0], cfg.PASSA_BANDA[1], picks="all",
               fir_design="firwin", skip_by_annotation="edge", verbose=False)
    if float(raw.info["sfreq"]) != float(cfg.TAXA_IA):
        raw.resample(cfg.TAXA_IA, npad="auto", verbose=False)
    return raw


def janela_bruta(sinal, taxa, amostras=None):
    """(C, L) em Volts, a `taxa` Hz -> (C, amostras) pronto para a rede.

    Usado para tudo que não vem do MNE: gravação da Cyton, buffer ao vivo. O sinal é
    espelhado nas bordas antes de filtrar e o espelho é recortado depois — sem isso, o
    FIR de 0,5 Hz distorce as pontas de uma janela de 4 s, que é justamente onde o
    transiente da transição está.
    """
    import mne
    from scipy.signal import detrend
    mne.set_log_level("ERROR")

    amostras = amostras or cfg.AMOSTRAS
    pad = int(cfg.PAD_FILTRO_S * taxa)
    espelhado = np.pad(np.asarray(sinal, dtype=np.float64), ((0, 0), (pad, pad)), mode="reflect")

    info = mne.create_info(list(cfg.CANAIS), taxa, "eeg")
    raw = mne.io.RawArray(espelhado, info, verbose=False)
    raw.set_montage("standard_1005", on_missing="ignore")
    cadeia_mne(raw)

    x = raw.get_data()
    corte = int(round(pad * cfg.TAXA_IA / taxa))
    if corte > 0:
        x = x[:, corte:-corte]
    x = ajustar_tamanho(x, amostras)
    return detrend(x, axis=1, type=cfg.DETREND).astype(np.float32)


def ajustar_tamanho(x, alvo):
    """(C, L) -> (C, alvo). Mantém o fim e completa o começo, se faltar.

    A reamostragem 250 -> 128 raramente cai exatamente no comprimento pedido; a diferença
    é de uma ou duas amostras. Se faltar muito, é sinal de que a janela foi montada
    errado — daí o aviso.
    """
    C, L = x.shape
    if L == alvo:
        return x
    if L > alvo:
        return x[:, L - alvo:]
    falta = alvo - L
    if falta > 0.02 * alvo:
        raise ValueError(f"janela curta demais: {L} amostras para um alvo de {alvo}. "
                         f"Confira a taxa de amostragem e o tamanho do buffer.")
    return np.pad(x, ((0, 0), (falta, 0)), mode="reflect" if falta < L else "edge")


def ajustar_lote(X, alvo=None):
    """(n, C, L) -> (n, C, alvo). Mesma regra do ajustar_tamanho, aplicada ao lote."""
    alvo = alvo or cfg.AMOSTRAS
    if X.shape[2] == alvo:
        return X
    return np.stack([ajustar_tamanho(x, alvo) for x in X]).astype(np.float32)


def estatistica(X):
    """(mu, sd) por canal de um conjunto (n, C, T). Calculada SÓ no treino."""
    mu = X.mean(axis=(0, 2), keepdims=True).astype(np.float32)
    sd = (X.std(axis=(0, 2), keepdims=True) + 1e-8).astype(np.float32)
    return mu, sd


def normalizar(X, mu, sd):
    """z-score por canal com a estatística dada. Aceita (C,T) ou (n,C,T)."""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim == 2:
        X = X[None]
    return ((X - mu) / sd).astype(np.float32)


def normalizar_com_modelo(modelo, X):
    """Normaliza usando a estatística que veio no checkpoint."""
    mu, sd = getattr(modelo, "norm", (None, None))
    if mu is None:
        mu, sd = estatistica(np.asarray(X)[None] if np.ndim(X) == 2 else X)
    return normalizar(X, mu, sd)
