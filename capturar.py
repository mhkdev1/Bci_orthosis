# -*- coding: utf-8 -*-
"""
capturar.py — grava uma sessão nova com a Cyton.

    python capturar.py --ensaio                 # sem capacete: confere tempos e formato
    python capturar.py --sujeito S006 --run R1  # grava de verdade (~4 min)

O que a sessão faz: mostra uma dica na tela, alternando descanso e tarefa a cada 4,1 s,
e grava o EEG do começo ao fim sem interrupção.

------------------------------------------------------------------------------
DUAS DECISÕES DE PROTOCOLO QUE VALEM EXPLICAR
------------------------------------------------------------------------------
1. NADA É DESCARTADO. O stream é drenado continuamente e todo bloco vai para o arquivo.
   O único descarte da sessão acontece uma vez, logo após a estabilização, para tirar o
   transitório de ligar a placa.

   Isso não é detalhe: numa versão anterior deste projeto a captura salvava um arquivo
   por trial e limpava o buffer antes de cada um, o que abria um vão de ~1,44 s entre
   trials. Como as janelas de análise precisam de meio segmento ANTES do início da
   tarefa, e esse meio segmento caía justamente no vão, ele tinha que ser fabricado por
   espelhamento — 20% a 50% de cada janela de movimento era sinal inventado. Gravando
   contínuo o problema simplesmente não existe.

2. DESCANSO ENTRE TODAS AS TAREFAS, com a mesma duração delas. É a estrutura do
   eegmmidb, e é o que torna as gravações comparáveis com o PhysioNet: em ambos, o
   início de qualquer segmento é uma transição, e a mesma geometria de janela vale para
   os dois conjuntos.

------------------------------------------------------------------------------
SAÍDA
------------------------------------------------------------------------------
    dados/gravacoes/<sujeito>_<run>/
        continuo.csv   uma linha por amostra: índice, tempo, rótulo, C3, C4, Cz (µV)
        eventos.csv    uma linha por segmento: índice, código, amostra de início

O detrend é aplicado por JANELA DE ANÁLISE (cortes nos pontos médios entre eventos), não
em blocos de tamanho fixo. Assim cada janela recebe a sua própria reta e o degrau entre
retas cai na borda da janela, nunca no centro — que é onde está o transiente. Medido:
essa escolha preserva 92% do ritmo mu e 96% do beta, contra 84% e 92% de blocos fixos de
4 s. Blocos muito curtos são destrutivos: a cada 10 amostras, sobra 30% do mu.
"""
from __future__ import annotations

import argparse
import os
import random
import time

import numpy as np
import pandas as pd

import config as cfg

DICAS = {
    "RE": ("DESCANSE", "cross"),
    "LH": ("IMAGINE FECHAR A MÃO ESQUERDA", "left"),
    "RH": ("IMAGINE FECHAR A MÃO DIREITA", "right"),
}


class PlacaSimulada:
    """Placa sintética para `--ensaio`. Valida tempos, mapeamento e formato sem hardware."""

    def __init__(self, n_linhas, canais, linha_tempo, taxa):
        self.n_linhas, self.canais, self.tempo, self.taxa = n_linhas, canais, linha_tempo, float(taxa)
        self._t = None

    def prepare_session(self):
        pass

    def start_stream(self):
        self._t = time.time()

    def get_board_data(self):
        agora = time.time()
        n = int((agora - self._t) * self.taxa)
        if n <= 0:
            return np.zeros((self.n_linhas, 0))
        d = np.zeros((self.n_linhas, n))
        t = self._t + np.arange(n) / self.taxa
        for k, c in enumerate(self.canais):
            d[c] = 30.0 * np.sin(2 * np.pi * (9 + k) * (t - t[0])) + np.random.randn(n) * 5
        d[self.tempo] = t
        self._t += n / self.taxa
        return d

    def stop_stream(self):
        pass

    def release_session(self):
        pass


def sequencia(n_tarefas, semente=None):
    """Alterna descanso e tarefa, começando por descanso. Mãos balanceadas e embaralhadas."""
    if semente is not None:
        random.seed(semente)
    mãos = ["LH", "RH"] * (n_tarefas // 2 + 1)
    mãos = mãos[:n_tarefas]
    random.shuffle(mãos)
    seq = []
    for m in mãos:
        seq.append("RE")
        seq.append(m)
    return seq


def cortes_por_janela(inicios, n):
    """Pontos médios entre eventos = bordas das janelas de análise."""
    on = np.asarray(inicios, dtype=float)
    if len(on) < 2:
        return []
    return [int(c) for c in ((on[:-1] + on[1:]) / 2).astype(int) if 0 < c < n]


def detrend_por_janela(x, cortes):
    """Uma reta por janela de análise. Devolve (sinal, tamanho dos degraus nas bordas)."""
    from scipy.signal import detrend
    x = np.asarray(x, dtype=np.float64)
    if not cortes:
        return detrend(x, axis=0, type="linear"), None
    y = detrend(x, axis=0, type="linear", bp=cortes)
    return y, np.abs(y[cortes] - y[[c - 1 for c in cortes]])


def mostrar(janela, texto, seta):
    """Desenha a dica. Sem cv2, cai para texto no terminal."""
    if janela is None:
        print(f"    >>> {texto}")
        return
    import cv2
    h, w = 720, 1280
    img = np.zeros((h, w, 3), np.uint8)
    cx, cy = w // 2, h // 2
    if seta == "right":
        cv2.arrowedLine(img, (cx - 220, cy), (cx + 220, cy), (255, 255, 255), 22, tipLength=0.4)
    elif seta == "left":
        cv2.arrowedLine(img, (cx + 220, cy), (cx - 220, cy), (255, 255, 255), 22, tipLength=0.4)
    else:
        cv2.line(img, (cx - 70, cy), (cx + 70, cy), (255, 255, 255), 10)
        cv2.line(img, (cx, cy - 70), (cx, cy + 70), (255, 255, 255), 10)
    cv2.putText(img, texto, (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    cv2.imshow(janela, img)
    if cv2.waitKey(1) == 27:
        raise KeyboardInterrupt


def drenar(placa, janela, segundos, blocos, texto, seta):
    """Espera `segundos` drenando o stream o tempo todo. Nada é descartado."""
    fim = time.time() + segundos
    while True:
        resta = fim - time.time()
        if resta <= 0:
            break
        time.sleep(min(0.05, resta))
        mostrar(janela, texto, seta)
        pedaco = placa.get_board_data()
        if pedaco.shape[1]:
            blocos.append(pedaco)


def main():
    ap = argparse.ArgumentParser(description="Grava uma sessão contínua com a Cyton.")
    ap.add_argument("--sujeito", default="S001")
    ap.add_argument("--run", default="R1")
    ap.add_argument("--tarefas", type=int, default=cfg.N_TAREFAS)
    ap.add_argument("--porta", default=None)
    ap.add_argument("--ensaio", action="store_true", help="placa sintética, sem hardware")
    ap.add_argument("--sem-tela", action="store_true", help="dicas só no terminal")
    a = ap.parse_args()

    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds
    placa_id = BoardIds.CYTON_BOARD.value
    canais_eeg = BoardShim.get_eeg_channels(placa_id)
    linha_tempo = BoardShim.get_timestamp_channel(placa_id)
    taxa = float(cfg.TAXA_PLACA)

    if a.ensaio:
        placa = PlacaSimulada(BoardShim.get_num_rows(placa_id), canais_eeg, linha_tempo, taxa)
        print("[ENSAIO] placa sintética — nenhum dado real.")
    else:
        params = BrainFlowInputParams()
        params.serial_port = a.porta or cfg.PORTA_EEG
        placa = BoardShim(placa_id, params)
        placa.prepare_session()

    janela = None
    if not (a.sem_tela or a.ensaio):
        try:
            import cv2
            janela = "Motor Imagery"
            cv2.namedWindow(janela, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        except ImportError:
            print("[aviso] opencv-python não instalado; dicas só no terminal.")

    seq = sequencia(a.tarefas)
    total = sum(cfg.DUR_DESCANSO if c == "RE" else cfg.DUR_TAREFA for c in seq)
    print("=" * 70)
    print(f"  {a.sujeito} {a.run} | {a.tarefas} tarefas + {a.tarefas} descansos "
          f"= {len(seq)} segmentos | {total/60:.1f} min")
    print(f"  {taxa:g} Hz | canais {cfg.CANAIS} | segmentos de {cfg.DUR_TAREFA:g} s")
    print("  Stream contínuo: nada é descartado depois da estabilização. ESC encerra.")
    print("=" * 70)

    placa.start_stream()
    print(f"Estabilizando {cfg.PAUSA_ESTABILIZACAO}s...")
    time.sleep(cfg.PAUSA_ESTABILIZACAO)
    placa.get_board_data()          # único descarte da sessão

    blocos, eventos = [], []
    t_zero = time.time()
    try:
        for i, code in enumerate(seq, 1):
            texto, seta = DICAS[code]
            mostrar(janela, texto, seta)
            eventos.append({"trial_index": i, "code": code, "t_onset": time.time()})
            print(f"  [{i:2d}/{len(seq)}] {code}  {texto:32s} t={time.time()-t_zero:6.1f}s")
            drenar(placa, janela, cfg.DUR_DESCANSO if code == "RE" else cfg.DUR_TAREFA,
                   blocos, texto, seta)
        resto = placa.get_board_data()
        if resto.shape[1]:
            blocos.append(resto)
    except KeyboardInterrupt:
        print("\nInterrompido — salvando o que já foi gravado.")
    finally:
        placa.stop_stream()
        placa.release_session()
        if janela is not None:
            import cv2
            cv2.destroyAllWindows()

    if not blocos:
        raise SystemExit("Nenhuma amostra capturada.")
    bruto = np.concatenate(blocos, axis=1)
    tempos = bruto[linha_tempo, :].astype(np.float64)
    n = bruto.shape[1]

    print("-" * 70)
    print(f"Amostras: {n} | {n/taxa:.1f} s de sinal | taxa medida "
          f"{n/(tempos[-1]-tempos[0]):.1f} Hz")

    # O BrainFlow entrega os dados em rajadas e carimba o tempo na chegada, então mapear
    # o cue pelo timestamp mais próximo erra até meia rajada. Como a contagem de amostras
    # é confiável, a base de tempo vem dela.
    inicios = [max(0, min(n - 1, int(round((e["t_onset"] - tempos[0]) * taxa)))) for e in eventos]
    espac = np.diff(inicios) / taxa if len(inicios) > 1 else np.array([])
    if len(espac):
        print(f"Espaçamento entre eventos: mediana {np.median(espac):.3f} s "
              f"(nominal {cfg.DUR_TAREFA:g} s), desvio {espac.std():.3f} s")
        if np.abs(np.median(espac) - cfg.DUR_TAREFA) > 0.3:
            print("  [aviso] espaçamento fora do nominal — confira a carga da bateria e o link.")

    canais = np.stack([bruto[canais_eeg[cfg.MAPA_CANAIS[c]], :] for c in cfg.CANAIS], axis=1)
    limpo, degraus = detrend_por_janela(canais, cortes_por_janela(inicios, n))
    print(f"Detrend por janela de análise | std por canal "
          f"{np.round(canais.std(0), 1)} -> {np.round(limpo.std(0), 1)} µV")
    if degraus is not None and len(degraus):
        print(f"  degrau nas bordas: mediana {np.median(degraus):.1f} µV")

    rotulos = np.empty(n, dtype=object)
    rotulos[:] = ""
    for j, ini in enumerate(inicios):
        fim = inicios[j + 1] if j + 1 < len(inicios) else n
        rotulos[ini:fim] = eventos[j]["code"]

    pasta = os.path.join(cfg.DIR_GRAVACOES, f"{a.sujeito}_{a.run}")
    os.makedirs(pasta, exist_ok=True)
    saida = pd.DataFrame({"sample_index": np.arange(n), "timestamp": tempos, "label": rotulos})
    for k, c in enumerate(cfg.CANAIS):
        saida[f"{c}_uV"] = np.round(limpo[:, k], 1)
    saida.to_csv(os.path.join(pasta, "continuo.csv"), index=False)
    pd.DataFrame([{"trial_index": e["trial_index"], "code": e["code"],
                   "amostra_inicio": ini, "t_s": round(ini / taxa, 4)}
                  for e, ini in zip(eventos, inicios)]).to_csv(
        os.path.join(pasta, "eventos.csv"), index=False)

    print("-" * 70)
    print(f"Salvo em {pasta}")
    print(f"\nPara conferir:\n  python inferir_offline.py --gravacao {a.sujeito}_{a.run}")


if __name__ == "__main__":
    main()
