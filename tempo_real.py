# -*- coding: utf-8 -*-
"""
tempo_real.py — capacete ligado, órtese acionada pela intenção de movimento.

    python tempo_real.py                  # com a órtese
    python tempo_real.py --sem-ortese     # só imprime, para teste sem hardware
    python tempo_real.py --simular-ortese # imprime o que mandaria à órtese

O laço é simples: a cada `config.PASSO_S` segundos, pega os últimos 4,1 s do stream,
passa pela mesma cadeia do treino, classifica, e entrega a probabilidade ao `Decisor`,
que decide se manda fechar. Toda a lógica de acionamento está em `nucleo/servo.py`.

------------------------------------------------------------------------------
O QUE ESPERAR, E O QUE NÃO ESPERAR
------------------------------------------------------------------------------
Este é o ponto do projeto onde a diferença entre o número de validação e o uso real
aparece inteira, e vale ser explícito sobre isso antes de alguém colocar a mão na órtese.

A rede foi treinada com janelas centradas na transição descanso -> tarefa. Ao vivo, a
janela deslizante só reproduz essa geometria quando a transição cai no centro dela — ou
seja, cerca de 2 segundos depois de o usuário começar a imaginar o movimento. Antes
disso a janela é quase toda descanso; depois, o transiente sai pela borda e o que sobra
é imagética sustentada, que separa bem pior. A detecção tem, portanto, uma janela útil
de mais ou menos um segundo, e é por isso que o `Decisor` TRAVA o estado depois de
fechar em vez de exigir confirmação contínua.

Medido em 6 runs contínuos do PhysioNet (44 tentativas de mão direita, 740 s) com o
limiar padrão de 0,80: cerca de 43% das tentativas geram um comando, com um disparo
fora de hora a cada 53 segundos. Ou seja: mais da metade das tentativas não fecha a mão,
e a mão fecha sozinha aproximadamente uma vez por minuto. Baixar o limiar aumenta a
detecção e piora os disparos falsos na mesma medida.

Um sistema com esse desempenho serve para demonstração e pesquisa. Não serve para uso
assistivo sem supervisão, e a órtese deve estar sempre acessível para ser removida.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

import config as cfg
from nucleo import preproc, servo
from nucleo.modelo import carregar


def main():
    ap = argparse.ArgumentParser(description="Inferência ao vivo com acionamento da órtese.")
    ap.add_argument("--modelo", default=cfg.MODELO_PADRAO)
    ap.add_argument("--porta", default=None, help="porta da Cyton (padrão: config.PORTA_EEG)")
    ap.add_argument("--sem-ortese", action="store_true", help="não aciona nada, só imprime")
    ap.add_argument("--simular-ortese", action="store_true", help="imprime os comandos, sem serial")
    ap.add_argument("--passo", type=float, default=cfg.PASSO_S)
    ap.add_argument("--limiar", type=float, default=cfg.LIMIAR)
    ap.add_argument("--classe", default=cfg.CLASSE_ACIONA)
    a = ap.parse_args()

    from brainflow.board_shim import BoardShim, BrainFlowInputParams, BoardIds

    modelo, amostras, esquema = carregar(a.modelo)
    if amostras != cfg.AMOSTRAS:
        sys.exit(f"O modelo espera {amostras} amostras e o config monta {cfg.AMOSTRAS}.")
    if a.classe not in cfg.CLASSES.values():
        sys.exit(f"--classe deve ser uma de {sorted(cfg.CLASSES.values())}")

    params = BrainFlowInputParams()
    params.serial_port = a.porta or cfg.PORTA_EEG
    placa = BoardShim(BoardIds.CYTON_BOARD.value, params)
    canais_eeg = BoardShim.get_eeg_channels(BoardIds.CYTON_BOARD.value)
    linhas = [canais_eeg[cfg.MAPA_CANAIS[c]] for c in cfg.CANAIS]
    taxa = float(cfg.TAXA_PLACA)
    # O buffer é dimensionado a partir do que o MODELO espera, não de um número fixo:
    # assim ele acompanha automaticamente qualquer mudança de janela ou de taxa.
    largura = int(np.ceil(amostras / cfg.TAXA_IA * taxa))

    print("=" * 74)
    print(f"  modelo ...: {os.path.basename(a.modelo)} | {esquema.get('nome', '?')}")
    print(f"  janela ...: {largura} amostras @ {taxa:g} Hz  ->  {amostras} @ {cfg.TAXA_IA} Hz "
          f"({amostras/cfg.TAXA_IA:.2f} s)")
    print(f"  decisão ..: fecha em '{a.classe}' acima de {a.limiar:g} | passo {a.passo:g} s")
    print(f"  órtese ...: curso 0-{cfg.ANGULO_FECHADO}°, trava {cfg.SEGURA_S:g} s após fechar")
    print("=" * 74)

    dec = servo.Decisor(classe=a.classe, limiar=a.limiar)
    ortese = None
    if not a.sem_ortese:
        ortese = servo.Ortese(simular=a.simular_ortese)
        if not ortese.simular and not ortese.ping():
            print("  [aviso] o firmware não respondeu ao ping. Seguindo mesmo assim.")

    placa.prepare_session()
    placa.start_stream()
    print(f"Stream ligado. Estabilizando {cfg.PAUSA_ESTABILIZACAO}s... (Ctrl-C encerra)")
    time.sleep(cfg.PAUSA_ESTABILIZACAO)

    try:
        while True:
            time.sleep(a.passo)
            bruto = placa.get_current_board_data(largura)
            if bruto.shape[1] < largura:
                continue                            # buffer ainda enchendo
            sinal = np.stack([bruto[r, :] for r in linhas]) * 1e-6
            x = preproc.janela_bruta(sinal, taxa, amostras)
            xn = preproc.normalizar_com_modelo(modelo, x)
            with torch.no_grad():
                prob = torch.softmax(modelo(torch.from_numpy(xn).float()), dim=1).numpy()[0]

            rot, acao = dec.passo(prob, time.monotonic())
            probs = "  ".join(f"{cfg.CLASSES[i]}={prob[i]:.2f}" for i in sorted(cfg.CLASSES))
            marca = ""
            if acao == "FECHAR":
                marca = f"   >>> FECHA ({cfg.ANGULO_FECHADO}°)"
                if ortese:
                    ortese.fechar()
            elif acao == "ABRIR":
                marca = "   <<< abre"
                if ortese:
                    ortese.abrir()
            elif dec.travado_ate is not None:
                marca = "   [travado]"
            print(f"  {rot:<5s} | {probs}{marca}")
    except KeyboardInterrupt:
        print("\nEncerrando.")
    finally:
        placa.stop_stream()
        placa.release_session()
        if ortese:
            ortese.fechar_conexao()      # abre a mão antes de soltar a serial
        print("Sessão encerrada, órtese aberta.")


if __name__ == "__main__":
    main()
