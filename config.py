# -*- coding: utf-8 -*-
"""
config.py — todos os parâmetros do projeto, num arquivo só.

Nenhum script tem número solto: se você quer mudar alguma coisa, muda aqui. Os valores
abaixo são os que produziram o modelo em `modelos/cnn2d_transicao.pt`, e vários deles
foram escolhidos por medição — quando foi o caso, o número medido está no comentário.
"""
import os

RAIZ = os.path.dirname(os.path.abspath(__file__))

# =====================================================================
# 1) HARDWARE
# =====================================================================
PORTA_EEG = "COM3"          # dongle da Cyton. Linux: "/dev/ttyUSB0"
PORTA_SERVO = "COM4"        # ESP32 da órtese. "" pergunta qual usar
BAUD_SERVO = 115200
TAXA_PLACA = 250            # Cyton no dongle USB é 250 Hz fixo

# Quais eletrodos, e em que ordem. A ordem importa: é a mesma que a rede viu no treino.
CANAIS = ["C3", "C4", "Cz"]
# Índice de cada canal na lista de canais EEG que o BrainFlow devolve para a Cyton.
MAPA_CANAIS = {"C3": 0, "C4": 1, "Cz": 2}

# =====================================================================
# 2) PROTOCOLO
# =====================================================================
# Durações medidas nos EDFs do eegmmidb (1800 segmentos), não tiradas do artigo —
# a página do PhysioNet não publica os tempos por trial.
DUR_DESCANSO = 4.1          # anotação T0
DUR_TAREFA = 4.1            # anotações T1 (mão esquerda) e T2 (mão direita)
N_TAREFAS = 30              # trials de tarefa por sessão de captura
PAUSA_ESTABILIZACAO = 2     # segundos de descarte no início, antes do primeiro trial

# =====================================================================
# 3) PRÉ-PROCESSAMENTO
# =====================================================================
TAXA_IA = 128               # o EDF vem a 160 Hz e a Cyton a 250; ambos caem para 128
NOTCH = 60.0                # rede elétrica
PASSA_BANDA = (0.5, 35.0)
# O corte inferior foi varrido nesta arquitetura (platô das últimas 10 épocas):
#   0,5 Hz -> 72,0%  |  1 Hz -> 69,1%  |  2 Hz -> 68,5%  |  4 Hz -> 61,0%
# Quase toda a perda acontece já entre 0,5 e 1 Hz. O corte superior nunca mudou nada.

DETREND = "linear"          # remove offset e inclinação. Obrigatório, em todos os pontos.
PAD_FILTRO_S = 4.0          # espelho temporário nas bordas, só para o FIR de 0,5 Hz não
                            # distorcer a janela curta. É recortado antes do modelo ver.

# =====================================================================
# 4) JANELA — o coração do método
# =====================================================================
# Cada janela é centrada numa TRANSIÇÃO e leva metade de cada segmento vizinho:
#
#   centro em descanso -> tarefa   = MOVIMENTO
#   centro em tarefa -> descanso   = DESCANSO
#
#   descanso  |##############|
#   movimento                |##############|
#   ciclo     TTTTTTTTTTTTTTT...............TTTTTTTTTTTTTT
#                            ^ início da tarefa
#
# Como os segmentos duram o mesmo, as duas janelas ladrilham a linha do tempo: encostam
# uma na outra, sem sobreposição e sem buraco. Nenhum trecho de sinal recebe dois rótulos.
MEIA_JANELA = DUR_TAREFA / 2.0            # 2,05 s de cada lado
JANELA_S = DUR_TAREFA                     # 4,1 s no total
AMOSTRAS = int(round(JANELA_S * TAXA_IA)) + 1     # 526, contando as duas pontas

CLASSES = {0: "rest", 1: "left", 2: "right"}
CODIGO_CLASSE = {"RE": 0, "LH": 1, "RH": 2, "BEO": 0}
# T1 = mão esquerda, T2 = mão direita (runs 3,4,7,8,11,12 do eegmmidb)
ANOTACAO_CLASSE = {"T0": 0, "T1": 1, "T2": 2}

# =====================================================================
# 5) TREINO
# =====================================================================
RUNS_EDF = [3, 4, 7, 8, 11, 12]   # runs de mão: 3,7,11 movimento real; 4,8,12 imagética
SUJEITOS_RUINS = [88, 89, 92, 100]  # gravados a 128 Hz ou com run corrompido
EPOCAS = 35
LOTE = 16
LOTE_VALIDACAO = 64               # só afeta o pico de RAM na validação
TAXA_APRENDIZADO = 1e-3
FRACAO_VALIDACAO = 0.2
PESO_POR_CLASSE = True            # compensa o desbalanceamento (rest é ~50% dos dados)
SEMENTE = 40

AUGMENT = True
AUG_PROB = 0.5                    # probabilidade de aplicar CADA transformação
AUG_RUIDO = 0.10                  # desvio do ruído gaussiano, em unidades de z-score
AUG_ESCALA = (0.9, 1.1)           # escala de amplitude por janela
AUG_DESLOCAMENTO = 16             # deslocamento temporal máximo, em amostras

# =====================================================================
# 6) ÓRTESE
# =====================================================================
CLASSE_ACIONA = "right"           # qual classe fecha a órtese

# ---- Servo: os valores abaixo espelham o firmware gravado no ESP32 ----
# O firmware é INCREMENTAL e trabalha na escala do próprio servo, com o repouso no
# CENTRO do curso. Ver firmware_esp32/firmware_esp32.ino:
#
#   'L'        angle = max(-30, angle - passo)
#   'R'        angle = min(180, angle + passo)
#   'C'        angle = 90            <- repouso é 90°, não 0°
#   '0'..'9'   passo = dígito, com '0' virando 1  (então 1 a 9 graus)
#   e imprime "angle=<n>" a cada byte recebido
#
# Mudar qualquer um destes exige mudar o .ino junto — eles têm que concordar.
SERVO_REPOUSO = 90                # para onde o 'C' leva o servo
SERVO_AMPLITUDE = 30              # quanto sai do repouso ao fechar; é o "30°" da órtese
SERVO_PASSO_GRAUS = 3             # 1 a 9. Divide 30 exato -> 10 pulsos para fechar
SERVO_INTERVALO_S = 0.05          # espera entre pulsos, igual à do controle manual
SERVO_SENTIDO_FECHA = "R"         # 'R' ou 'L' — descubra qual fecha a SUA montagem

# Ângulos absolutos, na escala do firmware. Derivados, não edite direto.
ANGULO_ABERTO = SERVO_REPOUSO
ANGULO_FECHADO = (SERVO_REPOUSO + SERVO_AMPLITUDE if SERVO_SENTIDO_FECHA == "R"
                  else SERVO_REPOUSO - SERVO_AMPLITUDE)
# A amplitude de 30° é o limite mecânico confortável do dedo indicador na montagem
# usada — acima disso a haste força a articulação. Não aumente sem conferir o batente
# físico, com a órtese vazia.
LIMIAR = 0.50                     # probabilidade mínima para acionar
# Medido em 6 runs contínuos do PhysioNet (44 cues, 740 s), passo de 0,25 s:
#   limiar 0,60 -> 73% de detecção, 1 disparo falso a cada 16 s
#   limiar 0,70 -> 68%,             1 a cada 29 s
#   limiar 0,80 -> 43%,             1 a cada 53 s   <- escolhido, pelo lado seguro
# Numa órtese, disparo falso é a mão fechando sozinha; é o erro que machuca.

PASSO_S = 0.25                    # intervalo entre inferências no tempo real
SEGURA_S = 3.0                    # tempo travado depois de fechar
REFRATARIO_S = 2.0                # tempo morto depois de soltar, antes de poder fechar

# =====================================================================
# 7) CAMINHOS
# =====================================================================
DIR_PHYSIONET = os.path.join(RAIZ, "dados", "physionet")
DIR_GRAVACOES = os.path.join(RAIZ, "dados", "gravacoes")
DIR_MODELOS = os.path.join(RAIZ, "modelos")
DIR_FIGURAS = os.path.join(RAIZ, "figuras")
MODELO_PADRAO = os.path.join(DIR_MODELOS, "cnn2d_transicao.pt")

# =====================================================================
# 8) ARQUITETURA DA REDE
# =====================================================================
CNN_KERNELS = (5, 11, 15, 33, 65)   # escalas temporais; a 128 Hz cobrem 25,6 a 2,0 Hz
CNN_FILTROS = 20                    # filtros por escala
CNN_DEPTH = 3                       # filtros espaciais por mapa. Com 3 eletrodos, 3 é o
                                    # teto útil: não há mais direções independentes.
CNN_SAIDA = 32                      # mapas depois do pointwise
