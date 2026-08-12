# Relatório de execução

Como rodar este projeto, do zero até a órtese fechando pela intenção de movimento.

O [README.md](README.md) é o resumo. Este documento é o roteiro completo: o que cada
programa faz, em que ordem executar, o que esperar de saída em cada etapa, como ler os
números e o que fazer quando algo não funciona.

**Nada aqui exige hardware para começar.** Duas das três etapas — treino e inferência
offline — rodam com o que já está no repositório. O capacete e a órtese só entram na
seção 6.

---

## Sumário

1. [O que o sistema faz](#1-o-que-o-sistema-faz)
2. [O caminho do sinal](#2-o-caminho-do-sinal)
3. [Instalação](#3-instalação)
4. [Etapa 1 — inferência offline (sem hardware)](#4-etapa-1--inferência-offline-sem-hardware)
5. [Etapa 2 — treino](#5-etapa-2--treino)
6. [Etapa 3 — hardware](#6-etapa-3--hardware)
7. [Gravar uma sessão nova](#7-gravar-uma-sessão-nova)
8. [Referência de comandos](#8-referência-de-comandos)
9. [Parâmetros que importam](#9-parâmetros-que-importam)
10. [Formato dos dados](#10-formato-dos-dados)
11. [Como ler os resultados](#11-como-ler-os-resultados)
12. [Problemas comuns](#12-problemas-comuns)
13. [Limites conhecidos](#13-limites-conhecidos)

---

## 1. O que o sistema faz

Três eletrodos de EEG sobre o córtex motor (**C3**, **C4**, **Cz**) alimentam uma rede
convolucional que classifica cada janela de 4,1 s em **descanso**, **mão esquerda** ou
**mão direita**. Quando a classe configurada em `CLASSE_ACIONA` aparece com probabilidade
acima do limiar, um ESP32 recebe a ordem de fechar a órtese de mão.

| peça | onde vive | papel |
|---|---|---|
| OpenBCI Cyton | hardware | lê o EEG a 250 Hz |
| `nucleo/preproc.py` | PC | notch, passa-banda, reamostragem para 128 Hz, detrend |
| `nucleo/modelo.py` | PC | a CNN 2D multiescala |
| `nucleo/servo.py` | PC | decide *quando* acionar e fala com o ESP32 |
| `firmware_esp32.ino` | dentro do ESP32 | recebe bytes e move o servo |

O elemento central do método está no `config.py`, seção 4: **cada janela é centrada numa
transição**, e leva metade de cada segmento vizinho.

```
descanso  |##############|
movimento                |##############|
                         ^ início da tarefa = centro da janela
```

Centro em `descanso → tarefa` é rotulado **movimento**; centro em `tarefa → descanso` é
rotulado **descanso**. Como os segmentos têm a mesma duração, as janelas ladrilham a linha
do tempo sem sobreposição e sem buraco — nenhum trecho de sinal recebe dois rótulos.

Esse detalhe é o que explica quase tudo que vem depois: a latência de ~2 s no tempo real, a
janela útil de detecção de cerca de um segundo, e a razão de o `Decisor` travar o estado
depois de fechar em vez de exigir confirmação contínua.

---

## 2. O caminho do sinal

O mesmo caminho vale para as três origens — EDF do PhysioNet, gravação da Cyton e stream ao
vivo. É essa igualdade que permite treinar com uma origem e inferir em outra.

```
       EDF 160 Hz          CSV 250 Hz          stream 250 Hz
    (dados/physionet)   (dados/gravacoes)      (Cyton ao vivo)
            |                   |                     |
            +---------+---------+----------+----------+
                      |                    |
              nucleo/dados.py        preproc.janela_bruta()
                      |                    |
                      +---------+----------+
                                |
                       nucleo/preproc.py
              notch 60 Hz -> passa-banda 0,5-35 Hz
              -> reamostra para 128 Hz -> detrend linear
              -> normaliza (mu/sd gravados no checkpoint)
                                |
                     janela (3 canais, 526 amostras)
                                |
                       nucleo/modelo.py  CNN2D
                    5 escalas (5,11,15,33,65) x 20 filtros
                                |
                    prob(rest) prob(left) prob(right)
                                |
                       nucleo/servo.py  Decisor
                    limiar -> trava 3 s -> refratário 2 s
                                |
                      Ortese: pulsos 'R'/'L'/'C'
                                |
                            ESP32 -> servo
```

Duas coisas valem registrar:

- **A janela tem 526 amostras**, não 525. `4,1 s x 128 Hz = 524,8`, que não é inteiro; o
  `mne.Epochs` devolve 525 e o `preproc.ajustar_lote()` completa a diferença, para que EDF e
  gravação cheguem à rede com exatamente o mesmo comprimento.
- **A normalização vem do checkpoint, não do config.** O `.pt` guarda `norm_mu`, `norm_sd`,
  a arquitetura e a geometria da janela. É o que evita o modo de falha clássico deste tipo
  de projeto: mudar um parâmetro meses depois e passar a alimentar o modelo com uma janela
  diferente da que ele aprendeu, sem erro nenhum aparecer.

---

## 3. Instalação

Python **3.9 ou mais novo**. Não é preciso GPU: tudo roda em CPU.

```bash
pip install -r requirements.txt
python verificar.py
```

### Sempre comece pelo `verificar.py`

Ele imprime qual interpretador está em uso, quais bibliotecas faltam, quantos arquivos de
dados foram encontrados, e monta a linha de instalação correta para *aquele* Python.

```
Python 3.10.11
  C:\Python310\python.exe

Dependências do treino e da inferência offline:
    ok   numpy
    ok   scipy
    ...
Dados:
    ok   PhysioNet  60 arquivo(s) em dados\physionet
    ok   gravações  2 arquivo(s) em dados\gravacoes
    ok   modelo     1 arquivo(s) em modelos

Tudo certo para treinar e inferir offline.
```

### Se você tem mais de um Python instalado

Situação comum no Windows: o `python` do PATH não é o mesmo que a sua IDE usa. O sintoma é
um `ModuleNotFoundError: No module named 'torch'` na primeira linha, mesmo tendo instalado
tudo. A solução é usar o caminho completo do **mesmo** interpretador nas duas operações:

```powershell
& "C:\caminho\para\python.exe" -m pip install -r requirements.txt
& "C:\caminho\para\python.exe" verificar.py
& "C:\caminho\para\python.exe" inferir_offline.py --sujeito S001
```

### O que cada dependência serve

| pacote | necessário para |
|---|---|
| `numpy`, `scipy`, `torch`, `mne`, `pandas`, `matplotlib` | treino e inferência offline — **o mínimo** |
| `brainflow` | falar com a Cyton (`capturar.py`, `tempo_real.py`) |
| `pyserial` | falar com o ESP32 da órtese |
| `opencv-python` | tela de dicas durante a captura (sem ela, as dicas vão para o terminal) |
| `pynput` | controle manual da órtese pelo teclado (`controle_servo_1.py`) |

Se você só quer reproduzir os resultados, os seis primeiros bastam.

---

## 4. Etapa 1 — inferência offline (sem hardware)

Esta é a etapa por onde começar. Ela usa o modelo já treinado (`modelos/cnn2d_transicao.pt`)
e os dados que vêm no repositório. Não precisa de capacete, órtese, nem treino.

### 4.1 Ver o que existe em disco

```bash
python inferir_offline.py --listar
```

```
PhysioNet (10 sujeitos em dados/physionet/):
  S001  runs [3, 4, 7, 8, 11, 12]
  S002  runs [3, 4, 7, 8, 11, 12]
  ...
Gravações da Cyton (em dados/gravacoes/):
  S004_R1
  S005_R1_cortado
```

### 4.2 Rodar um sujeito do PhysioNet

```bash
python inferir_offline.py --sujeito S001
python inferir_offline.py --sujeito S001 --run 4     # um run só
```

A saída tem quatro blocos: cabeçalho (fonte, nº de janelas, modelo, esquema), tabela de
precisão/recall/F1 por classe, matriz de confusão, e uma nota de como ler.

```
============================================================================
  RESULTADO POR JANELA
============================================================================
    classe |  precisão   recall       F1 |     n
  -------------------------------------------------
      rest |     78.2%    81.4%    0.798 |    86
      left |     69.1%    64.3%    0.666 |    42
     right |     71.0%    69.0%    0.700 |    42
  -------------------------------------------------
  macro-F1 |                       0.723 |   170

  acurácia .......: 125/170 = 73.6%
  macro-F1 .......: 0.723
  linha de base ..: 0.436  (responder sempre 'rest' daria 50.6%)
  veredito .......: acima da linha de base
```

### 4.3 Rodar uma gravação da Cyton

```bash
python inferir_offline.py --gravacao S005_R1_cortado
python inferir_offline.py --gravacao S004_R1 --detalhe    # imprime janela a janela
```

Com as gravações contínuas aparece um **quinto bloco**, que não existe nos EDFs: a varredura
por janela deslizante.

```
============================================================================
  JANELA DESLIZANTE (o que a órtese enfrentaria)
============================================================================
  passo 0.25 s | limiar 0.5 | classe 'right' | 193 s
  comandos de fechar enviados ..: 14
  cues de 'right' detectados : 9/15 | latência mediana 2.10 s
  disparos fora de cue .........: 5 (1 a cada 39 s)
```

**As duas leituras respondem perguntas diferentes**, e essa distinção é o ponto mais
importante do relatório:

| leitura | o que faz | o que responde |
|---|---|---|
| **por janela** | recorta na mesma geometria do treino, alinhado ao evento | "o modelo aprendeu?" — é o número comparável à validação |
| **janela deslizante** | varre o sinal inteiro sem saber onde estão os eventos | "o sistema serve na prática?" |

A segunda é sempre bem pior que a primeira. A diferença entre elas é o custo de não ter um
cue para se alinhar, e é ela que decide se o sistema funciona fora do laboratório.

A latência mediana fica perto de 2,05 s (`MEIA_JANELA`) porque é aí que a janela deslizante
reproduz a geometria de treino: a transição no centro.

### 4.4 Inspecionar um trial isolado

Útil para entender um erro específico, ou para demonstração.

```bash
python inferir_offline.py --sujeito S001 --trials              # lista os trials
python inferir_offline.py --sujeito S001 --trial right 5       # roda o 5º 'right'
```

O índice é **dentro da classe**, contando de 1 — "o quinto movimento de mão direita", não
"a janela número 23". A saída mostra as três probabilidades em barras e diz se acertou:

```
  o que era ....: right
  o que o modelo respondeu: right

     rest  0.118 |####                                    |
     left  0.094 |####                                    |
    right  0.788 |###############################         | <- resposta

  ACERTOU. Confiança 78.8% — acima do limiar de 0.5, acionaria a órtese.
```

### 4.5 Ver os comandos que a órtese receberia

Ainda sem hardware nenhum:

```bash
python inferir_offline.py --gravacao S005_R1_cortado --ortese --simular-ortese
```

`--ortese` liga o acionamento durante a varredura; `--simular-ortese` faz o `nucleo/servo.py`
imprimir os comandos em vez de abrir a serial. Com a órtese de verdade ligada, tire o
`--simular-ortese` e leia a seção 6.

> `--ortese` só tem efeito com `--gravacao`. A varredura contínua precisa de sinal sem
> cortes, e os EDFs entram no programa já recortados em janelas — o programa avisa se você
> pedir isso com `--sujeito`.

---

## 5. Etapa 2 — treino

### 5.1 Conferir o dataset antes de gastar tempo

```bash
python treinar.py --conferir
```

Monta o dataset inteiro, imprime as contas e **para**, sem treinar. Use sempre isso antes de
um treino longo — é onde você descobre que uma gravação não foi lida ou que um sujeito ficou
de fora.

```
PhysioNet: 60 arquivos de 10 sujeitos
  ...20/60
  ...40/60
  ...60/60
Gravação S004_R1: 59 janelas
Gravação S005_R1_cortado: 46 janelas

Dataset: 1805 janelas | {'rest': 902, 'left': 451, 'right': 452}
  PhysioNet: 1700 | gravações: 105
  X(1805, 3, 526)
OK — rode sem --conferir para treinar.
```

> A contagem de janelas de uma gravação é sempre **uma a menos** que a de eventos: a janela
> do primeiro evento precisaria de metade de um segmento anterior que não existe, e é
> descartada.

### 5.2 Treinar

```bash
python treinar.py                    # PhysioNet + gravações, 35 épocas
python treinar.py --sujeitos 5       # subconjunto, para um teste rápido
python treinar.py --sem-gravacoes    # só PhysioNet
python treinar.py --so-gravacoes     # só as gravações da Cyton
```

Roda na CPU. Com os 10 sujeitos deste repositório leva alguns minutos; o modelo distribuído
foi treinado com 105 sujeitos e levou cerca de duas horas.

A cada época sai uma linha, e no fim o relatório da melhor época:

```
Treinando 35 épocas na CPU | treino=1444 validação=361
--------------------------------------------------------------------------
  época   1/35 | loss 1.0421 | treino  44.2% | val_loss 1.0102 | validação  48.5% |  12s
  época   2/35 | loss 0.9887 | treino  51.7% | val_loss 0.9640 | validação  54.3% |  12s
  ...
--------------------------------------------------------------------------
Melhor validação: 72.0% (época 31)
```

O modelo salvo é o da **melhor época de validação**, não o da última.

### 5.3 Onde o modelo vai parar

Por padrão em `modelos/cnn2d_novo.pt` — **o modelo distribuído não é sobrescrito**. Para
usar o novo:

```bash
python inferir_offline.py --modelo modelos/cnn2d_novo.pt --sujeito S001
```

Para continuar de um checkpoint existente (o original continua intacto):

```bash
python treinar.py --continuar-de modelos/cnn2d_transicao.pt --so-gravacoes
```

Esse comando é o ajuste fino ao usuário: parte do modelo genérico e especializa nas
gravações de uma pessoa.

### 5.4 Flags úteis

| flag | efeito |
|---|---|
| `--epocas N` | número de épocas (padrão `config.EPOCAS` = 35) |
| `--lote N` | tamanho do lote (padrão 16) |
| `--lr X` | taxa de aprendizado (padrão 1e-3) |
| `--saida CAMINHO` | onde gravar o `.pt` |
| `--sujeitos N` | usa só os N primeiros sujeitos |
| `--conferir` | monta o dataset e para |

### 5.5 Uma ressalva sobre o número de validação

A separação treino/validação é estratificada **por classe, não por sujeito**. Janelas do
mesmo sujeito caem dos dois lados, então a acurácia de validação é otimista para o caso
"sujeito novo" — que é justamente o caso de quem coloca o capacete pela primeira vez.

Para medir isso de verdade: treine deixando um sujeito de fora e infira só nele.

---

## 6. Etapa 3 — hardware

Aqui entram o capacete e a órtese. **Faça as subseções na ordem.**

### 6.1 Montagem dos eletrodos

Três canais no sistema 10-20, sobre o córtex motor:

| canal | posição | o que capta |
|---|---|---|
| C3 | hemisfério esquerdo | movimento da mão **direita** |
| C4 | hemisfério direito | movimento da mão **esquerda** |
| Cz | linha média | referência de atividade motora central |

A ordem em `config.CANAIS` precisa bater com a ordem física na Cyton. É a mesma ordem que a
rede viu no treino, e **trocá-la degrada o resultado sem gerar erro nenhum**.

### 6.2 Gravar o firmware no ESP32

Abra `firmware_esp32/firmware_esp32.ino` na IDE do Arduino, instale a biblioteca
**ESP32Servo** pelo Gerenciador de Bibliotecas, selecione a sua placa e grave.

Ligação:

```
servo (sinal)  ->  GPIO 18
servo (VCC)    ->  fonte externa 5 V   (NÃO use o 5V do ESP32)
servo (GND)    ->  GND comum com o ESP32
```

O GND precisa ser comum, senão o servo treme ou não responde. Alimentar o servo pelo
regulador da placa costuma derrubá-la no pico de corrente da partida.

### 6.3 O protocolo do servo

O ESP32 recebe **um byte por comando, sem terminador**, e o movimento é **incremental**:

| byte | efeito |
|---|---|
| `R` | `angle = min(180, angle + passo)` |
| `L` | `angle = max(-30, angle - passo)` |
| `C` | `angle = 90` — o repouso é o **centro** do curso, não zero |
| `0`–`9` | tamanho do passo em graus; o dígito **é** o passo (`0` vira 1) |

A cada byte o firmware imprime `angle=<n>`. O lado do PC lê esse eco e o usa como posição
real, em vez de contar pulsos às cegas — e **precisa lê-lo de qualquer forma**, senão a saída
se acumula no buffer do host durante a sessão inteira.

Três consequências que o `nucleo/servo.py` trata explicitamente:

1. **O repouso é 90°, não 0°.** "Fechar 30°" quer dizer ir de 90 para 120 (ou para 60, se a
   sua montagem fechar no outro sentido). Por isso `SERVO_AMPLITUDE` existe separado do
   ângulo absoluto.
2. **O dígito é o passo, não passo−1.** Mandar `3` dá passos de 3°. Como o máximo é `9`, não
   existe passo de 10°.
3. **A posição é ressincronizada a cada `C`.** Toda sequência de fechamento parte do repouso,
   e o programa devolve a órtese ao repouso ao terminar, inclusive se for interrompido. Sem
   isso um pulso perdido faria o erro acumular e o limite de 30° deixaria de valer.

### 6.4 Testar a comunicação — primeiro sem hardware

```bash
python testar_ortese.py
```

Este teste **emula o ESP32 em software**, reimplementando a lógica do `.ino` byte a byte. Se
o PC e o firmware discordarem em alguma regra, ele falha aqui — sem hardware e sem risco de
forçar o dedo de alguém. Cinco verificações:

1. o dígito do passo é interpretado igual dos dois lados
2. `C` leva ao repouso e o PC reconhece o valor
3. fechar chega exatamente à amplitude configurada
4. o PC nunca manda o servo além da amplitude, mesmo pedindo mais
5. a cadeia decisor → órtese produz a sequência certa de comandos

```
--- 3. o limite segura ---------------------------------------------
  ok   pedir 60° a mais não move nada
  ok   o servo não passou do limite
...
  Tudo certo. O PC e o firmware concordam em todas as regras testadas.
```

Depois, **com a órtese ligada e VAZIA**:

```bash
python testar_ortese.py --real
python testar_ortese.py --real --detalhe     # mostra cada byte trocado
```

### 6.5 Descobrir as portas seriais

```bash
python -c "import serial.tools.list_ports as p; [print(x.device,'-',x.description) for x in p.comports()]"
```

E ajuste no `config.py`:

```python
PORTA_EEG   = "COM3"     # dongle da Cyton      (Linux: /dev/ttyUSB0)
PORTA_SERVO = "COM4"     # ESP32 da órtese      ("" faz o programa perguntar)
```

### 6.6 Descobrir qual sentido fecha a sua montagem

```bash
python controle_servo_1.py
```

Setas **←** e **→** movem, **↑** volta ao repouso, dígitos mudam o passo, **ESC** sai. Veja
qual letra fecha a sua órtese e ajuste:

```python
SERVO_SENTIDO_FECHA = "R"    # 'R' ou 'L'
SERVO_PASSO_GRAUS   = 3      # 1 a 9 graus por pulso
SERVO_AMPLITUDE     = 30     # curso máximo, em graus a partir do repouso
```

> `controle_servo_1.py` tem a porta serial **fixa no topo do próprio arquivo**
> (`SERIAL_PORT = 'COM3'`), não no `config.py`. É o único script assim — ajuste ali.

**O curso é limitado a 30°.** É o limite mecânico confortável da montagem usada: acima disso
a haste força a articulação do dedo. O módulo nunca envia pulsos além do teto, e trunca para
baixo quando o ângulo não é múltiplo do passo — pedir 30° com passo de 4° para em 28°, nunca
em 32°. Com o passo padrão de 3°, os 10 pulsos chegam exatos.

### 6.7 Testar o capacete sozinho

```bash
python tempo_real.py --sem-ortese
```

Nada é acionado; o programa só imprime a classificação a cada 0,25 s. É assim que você
confere impedância, contato dos eletrodos e ordem dos canais antes de plugar a órtese.

```
==========================================================================
  modelo ...: cnn2d_transicao.pt | centrado-transicao-v2
  janela ...: 1028 amostras @ 250 Hz  ->  526 @ 128 Hz (4.11 s)
  decisão ..: fecha em 'right' acima de 0.5 | passo 0.25 s
==========================================================================
Stream ligado. Estabilizando 2s... (Ctrl-C encerra)
  rest  | rest=0.71  left=0.15  right=0.14
  rest  | rest=0.63  left=0.19  right=0.18
  right | rest=0.21  left=0.13  right=0.66   >>> FECHA (120°)
  right | rest=0.24  left=0.15  right=0.61   [travado]
```

### 6.8 Rodar completo

```bash
python tempo_real.py --simular-ortese   # capacete real, órtese simulada
python tempo_real.py                    # capacete + órtese
```

Flags de ajuste sem editar o config:

| flag | efeito |
|---|---|
| `--limiar X` | probabilidade mínima para acionar |
| `--classe rest\|left\|right` | qual classe fecha a órtese |
| `--passo X` | segundos entre inferências |
| `--porta COMn` | porta da Cyton |
| `--modelo CAMINHO` | outro checkpoint |

**Ctrl-C encerra com segurança**: o `finally` para o stream, libera a sessão e devolve a
órtese ao repouso **antes** de soltar a serial. Se o processo terminar com a órtese fechada,
ela fica fechada no dedo de alguém até o próximo comando — daí o cuidado.

### 6.9 A lógica de acionamento

Está toda em `nucleo/servo.py`, na classe `Decisor`. Três mecanismos:

1. **Limiar** — a janela só conta como movimento se a probabilidade passar de
   `config.LIMIAR`. Abaixo disso é tratada como descanso.
2. **Trava** (`SEGURA_S = 3 s`) — depois de fechar, o estado fica travado sem reavaliar.
   Isso não é conveniência, é necessidade: a evidência que a rede usa é o transiente da
   transição, e ele some da janela deslizante poucos segundos depois do início do movimento.
   Exigir confirmação contínua faria a mão reabrir sozinha enquanto o usuário ainda está
   tentando fechá-la.
3. **Refratário** (`REFRATARIO_S = 2 s`) — tempo morto depois de soltar, antes de poder
   acionar de novo, para a órtese não ficar batendo.

A votação por maioria (M de N janelas) foi testada e **descartada**: medida em 6 runs
contínuos, ela derruba a detecção junto com o ruído, porque o pico de evidência dura cerca de
uma janela e os disparos falsos também são isolados. Exigir 3 de 4 zerava as duas coisas. O
parâmetro `votos` continua exposto para quem quiser re-testar com outro modelo.

---

## 7. Gravar uma sessão nova

```bash
python capturar.py --ensaio                  # placa sintética, sem capacete
python capturar.py --sujeito S006 --run R1   # ~4 minutos
```

Rode o `--ensaio` primeiro: ele usa uma placa sintética e valida os tempos, o mapeamento dos
canais e o formato de saída sem nenhum hardware.

A sessão mostra uma dica na tela (seta para a esquerda, seta para a direita, ou cruz de
fixação), alternando descanso e tarefa a cada 4,1 s. Padrão: 30 tarefas + 30 descansos = 60
segmentos, ≈4,1 min. **ESC encerra** e salva o que já foi gravado.

| flag | efeito |
|---|---|
| `--tarefas N` | número de tarefas (padrão 30) |
| `--sem-tela` | dicas só no terminal, sem janela do OpenCV |
| `--porta COMn` | porta da Cyton |

Duas decisões de protocolo que valem explicar:

**Nada é descartado.** O stream é drenado continuamente e todo bloco vai para o arquivo. O
único descarte acontece uma vez, logo após a estabilização, para tirar o transitório de ligar
a placa. Numa versão anterior a captura salvava um arquivo por trial e limpava o buffer antes
de cada um, o que abria um vão de ~1,44 s entre trials — e como as janelas precisam de meio
segmento *antes* do início da tarefa, esse meio segmento caía justamente no vão e tinha que
ser fabricado por espelhamento. De 20% a 50% de cada janela de movimento era sinal inventado.
Gravando contínuo, o problema não existe.

**Descanso entre todas as tarefas, com a mesma duração delas.** É a estrutura do eegmmidb, e
é o que torna as gravações comparáveis com o PhysioNet: em ambos, o início de qualquer
segmento é uma transição.

A gravação vai para `dados/gravacoes/S006_R1/` e já pode ser usada no treino e na inferência:

```bash
python inferir_offline.py --gravacao S006_R1
```

---

## 8. Referência de comandos

### Sem hardware nenhum

```bash
python verificar.py                                   # confere o ambiente
python testar_ortese.py                               # valida a lógica da órtese (emulada)
python capturar.py --ensaio                           # valida o protocolo de captura
python inferir_offline.py --listar                    # o que existe em disco
python inferir_offline.py --sujeito S001              # acurácia num sujeito
python inferir_offline.py --gravacao S004_R1          # acurácia numa gravação
python inferir_offline.py --sujeito S001 --trials     # lista os trials
python inferir_offline.py --sujeito S001 --trial right 5
python inferir_offline.py --gravacao S004_R1 --ortese --simular-ortese
python treinar.py --conferir                          # monta o dataset e para
python treinar.py --sujeitos 5                        # treino curto
```

### Só com a órtese

```bash
python testar_ortese.py --real                        # com a órtese VAZIA
python controle_servo_1.py                            # controle manual pelo teclado
python inferir_offline.py --gravacao S004_R1 --ortese --ritmo
```

`--ritmo` faz a varredura andar no tempo do **sinal**, não na velocidade da CPU. Sem isso ela
corre várias vezes mais rápido que a gravação e a órtese recebe os comandos amontoados —
inútil para assistir e ruim para o servo, que não acompanha. Com `--ritmo`, a varredura leva
o tempo real da gravação (3 a 4 min).

### Com capacete

```bash
python tempo_real.py --sem-ortese                     # só imprime
python capturar.py --sujeito S006 --run R1            # grava uma sessão
```

### Tudo ligado

```bash
python tempo_real.py
```

---

## 9. Parâmetros que importam

Tudo fica em [config.py](config.py). Nenhum script tem número solto — se quiser mudar algo,
mude ali. Estes são os que mais afetam a execução:

### Hardware

| parâmetro | padrão | o que faz |
|---|---|---|
| `PORTA_EEG` | `"COM3"` | dongle da Cyton (Linux: `/dev/ttyUSB0`) |
| `PORTA_SERVO` | `"COM4"` | ESP32 da órtese; `""` faz o programa perguntar |
| `CANAIS` | `["C3","C4","Cz"]` | **a ordem tem que bater com a física** |
| `MAPA_CANAIS` | `{C3:0, C4:1, Cz:2}` | índice de cada canal na lista do BrainFlow |

### Órtese

| parâmetro | padrão | o que faz |
|---|---|---|
| `CLASSE_ACIONA` | `"right"` | qual classe fecha a órtese |
| `LIMIAR` | `0.50` | probabilidade mínima para acionar |
| `SERVO_REPOUSO` | `90` | para onde o `C` leva o servo |
| `SERVO_AMPLITUDE` | `30` | quanto sai do repouso ao fechar |
| `SERVO_PASSO_GRAUS` | `3` | 1 a 9; divide 30 exato → 10 pulsos |
| `SERVO_SENTIDO_FECHA` | `"R"` | qual letra fecha a **sua** montagem |
| `SEGURA_S` | `3.0` | tempo travado depois de fechar |
| `REFRATARIO_S` | `2.0` | tempo morto depois de soltar |
| `PASSO_S` | `0.25` | intervalo entre inferências no tempo real |

**O limiar é o compromisso central do sistema.** Medido em 6 runs contínuos do PhysioNet
(44 cues, 740 s, passo de 0,25 s):

| limiar | detecção | disparos falsos |
|---|---|---|
| 0,60 | 73% | 1 a cada 16 s |
| 0,70 | 68% | 1 a cada 29 s |
| 0,80 | 43% | 1 a cada 53 s |

Numa órtese, disparo falso é a mão fechando sozinha — é o erro que machuca. Baixar o limiar
aumenta a detecção e piora os disparos falsos na mesma medida.

> **Atenção:** o `config.py` está entregue com `LIMIAR = 0.50`, que é mais permissivo que
> qualquer linha da tabela acima. Os comentários do código e o texto do
> [README.md](README.md) descrevem 0,80 como o valor "escolhido, pelo lado seguro". Para
> reproduzir os números de segurança citados, use `LIMIAR = 0.80` — ou passe
> `--limiar 0.8` no `tempo_real.py`, que não exige editar arquivo.

### Pré-processamento e janela

| parâmetro | padrão | observação |
|---|---|---|
| `TAXA_IA` | `128` | EDF vem a 160 Hz e Cyton a 250; ambos caem para 128 |
| `PASSA_BANDA` | `(0.5, 35.0)` | o corte inferior foi varrido: 0,5 Hz → 72,0%; 1 Hz → 69,1%; 2 Hz → 68,5%; 4 Hz → 61,0% |
| `NOTCH` | `60.0` | rede elétrica — mude para 50 onde a rede for 50 Hz |
| `JANELA_S` | `4.1` | = `DUR_TAREFA`, centrada na transição |
| `AMOSTRAS` | `526` | derivado; **não edite direto** |

Mudar `JANELA_S` invalida o modelo treinado. Os programas detectam isso e param com uma
mensagem clara em vez de rodar errado:

```
O modelo espera 526 amostras e o config monta 512.
Alinhe config.JANELA_S com o modelo, ou use outro checkpoint.
```

### Treino

| parâmetro | padrão |
|---|---|
| `EPOCAS` | `35` |
| `LOTE` | `16` |
| `TAXA_APRENDIZADO` | `1e-3` |
| `FRACAO_VALIDACAO` | `0.2` |
| `SEMENTE` | `40` |
| `AUGMENT` | `True` (ruído 0,10, escala 0,9–1,1, deslocamento ±16 amostras) |
| `SUJEITOS_RUINS` | `[88, 89, 92, 100]` — gravados a 128 Hz ou com run corrompido |

---

## 10. Formato dos dados

### PhysioNet — `dados/physionet/Sxxx/SxxxRyy.edf`

Dez sujeitos do **EEG Motor Movement/Imagery Dataset (eegmmidb)**, seis runs cada: **3, 7,
11** (movimento real da mão) e **4, 8, 12** (imagética). Base pública, disponível em
<https://physionet.org/content/eegmmidb/1.0.0/>.

Anotações: `T0` = descanso, `T1` = mão esquerda, `T2` = mão direita.

O modelo distribuído foi treinado com 105 sujeitos; os dez aqui bastam para reproduzir o
pipeline e rodar um treino de demonstração.

### Gravações próprias — `dados/gravacoes/<sessão>/`

Duas sessões com a OpenBCI Cyton, três eletrodos, 250 Hz.

| sessão | duração | segmentos | observação |
|---|---|---|---|
| `S004_R1` | 248 s | 60 | sessão completa |
| `S005_R1_cortado` | 193 s | 47 | a bateria acabou no meio; a parte instável foi removida |

O corte do S005 não foi arbitrário: a partir do evento 48 os segmentos passaram a mapear para
a mesma amostra, sinal de que a placa parou de transmitir. Tudo até ali está íntegro.

Cada pasta tem dois arquivos:

**`continuo.csv`** — uma linha por amostra:

```
sample_index,timestamp,label,C3_uV,C4_uV,Cz_uV
0,1754952821.32,RE,-12.4,8.1,3.7
1,1754952821.324,RE,-11.9,7.8,4.0
```

**`eventos.csv`** — uma linha por segmento:

```
trial_index,code,amostra_inicio,t_s
1,RE,0,0.0
2,LH,1025,4.1
3,RE,2050,8.2
```

Códigos: `RE` = descanso, `LH` = mão esquerda, `RH` = mão direita.

O detrend já vem aplicado no `continuo.csv`, **por janela de análise** (cortes nos pontos
médios entre eventos), não em blocos de tamanho fixo. Assim cada janela recebe a sua própria
reta e o degrau entre retas cai na borda da janela, nunca no centro — que é onde está o
transiente. Medido: essa escolha preserva 92% do ritmo mu e 96% do beta, contra 84% e 92% de
blocos fixos de 4 s. Blocos muito curtos são destrutivos: a cada 10 amostras, sobra 30% do mu.

### Modelo — `modelos/cnn2d_transicao.pt`

Dicionário do PyTorch com `state_dict`, `arch`, `n_canais`, `n_classes`, `amostras`,
`norm_mu`, `norm_sd` e `esquema`. É autossuficiente: a inferência reconstrói a mesma cadeia
sem depender do `config.py`.

---

## 11. Como ler os resultados

### Olhe o macro-F1, não a acurácia

Descanso é cerca de metade dos dados. Um modelo que responde sempre `rest` tira ~50% de
acurácia sem ter aprendido nada. O programa imprime a **linha de base** justamente para isso:

```
  linha de base ..: 0.436  (responder sempre 'rest' daria 50.6%)
  veredito .......: acima da linha de base
```

### Poucas janelas, pouca conclusão

Com algumas dezenas de janelas, diferença de alguns pontos percentuais não é significativa. O
próprio programa avisa no bloco final.

### O erro que importa é assimétrico

O relatório do `treinar.py` destaca uma linha específica:

```
Movimento classificado como descanso: 18/84 (21.4%) — é o erro que impede a órtese de fechar.
```

Movimento lido como descanso é **frustração**: a mão não fecha. Descanso lido como movimento
é **risco**: a mão fecha sozinha. São erros de peso muito diferente, e o limiar é o botão que
os troca um pelo outro.

### Resultados de referência

Os números abaixo saem direto dos comandos indicados, sem nenhum ajuste, com o modelo
distribuído:

| comando | acurácia | macro-F1 |
|---|---|---|
| `python inferir_offline.py --sujeito S001` | 73,6% | 0,723 |
| `python inferir_offline.py --gravacao S005_R1_cortado` | 66,7% | 0,634 |

O treino usa semente fixa (`config.SEMENTE = 40`), mas o resultado exato ainda pode variar um
pouco entre máquinas por causa de diferenças de BLAS e de versão do PyTorch.

---

## 12. Problemas comuns

| sintoma | causa provável | o que fazer |
|---|---|---|
| `ModuleNotFoundError: No module named 'torch'` | instalou num Python e está rodando em outro | `python verificar.py` mostra qual interpretador está em uso; instale com o caminho completo dele |
| `O modelo espera 526 amostras e o config monta N` | `JANELA_S`, `TAXA_IA` ou `DUR_TAREFA` foi alterado depois do treino | volte o config ao original, ou treine de novo com a geometria nova |
| `Nenhum dado. Confira dados/physionet/ e dados/gravacoes/` | pastas vazias ou nomes fora do padrão | os EDFs têm que se chamar `SxxxRyy.edf`; as gravações precisam de `eventos.csv` |
| `Sujeito S0xx não está em dados/physionet/` | sujeito não baixado | `python inferir_offline.py --listar` mostra o que existe |
| `Nenhuma porta serial encontrada` | ESP32 desconectado ou sem driver | confira o cabo; no Windows pode faltar o driver CP210x/CH340 |
| o firmware não respondeu ao `ping` | porta errada, ou baud diferente de 115200 | `python testar_ortese.py --real --detalhe` mostra cada byte trocado |
| o servo treme ou não responde | GND não comum, ou servo alimentado pelo ESP32 | GND comum e fonte externa de 5 V para o servo |
| a órtese fecha no sentido errado | `SERVO_SENTIDO_FECHA` invertido | rode `controle_servo_1.py`, veja qual letra fecha, e troque `"R"`↔`"L"` |
| a órtese não chega aos 30° | passo não divide a amplitude | o código trunca para baixo de propósito; use um passo que divida (3 divide 30) |
| a mão fecha sozinha o tempo todo | limiar baixo demais | suba `LIMIAR`, ou passe `--limiar 0.8` |
| a mão nunca fecha | limiar alto, ou eletrodos com contato ruim | rode `tempo_real.py --sem-ortese` e olhe as probabilidades brutas |
| `[aviso] espaçamento fora do nominal` na captura | bateria fraca ou link instável | recarregue a Cyton e regrave; foi o que aconteceu com o `S005` |
| memória estourando no treino | lote de validação grande demais | baixe `LOTE_VALIDACAO`; um forward único passa de 2 GB |
| a varredura com `--ortese` corre acelerada | falta `--ritmo` | acrescente `--ritmo` para andar no tempo da gravação |

---

## 13. Limites conhecidos

Esta seção existe para ser lida **antes** de colocar a órtese em alguém.

### O que a rede detecta

Não é contração muscular: é a **dessincronização sensório-motora (ERD)** sobre o córtex,
medida em C3/C4/Cz. Esse sinal indica *intenção* de mover, com latência de ~2 s, e não
carrega amplitude — não há como inferir dele "quanto" fechar. Um arco curto e fixo é o
comando honesto para um detector binário: sinaliza que a intenção foi reconhecida, sem fingir
uma proporcionalidade que a medida não tem.

### A janela útil de detecção dura cerca de um segundo

A rede foi treinada com janelas centradas na transição descanso → tarefa. Ao vivo, a janela
deslizante só reproduz essa geometria quando a transição cai no centro dela — ou seja, cerca
de 2 segundos depois de o usuário começar a imaginar o movimento. Antes disso a janela é
quase toda descanso; depois, o transiente sai pela borda e o que sobra é imagética
sustentada, que separa bem pior.

### O desempenho real

Medido em 6 runs contínuos do PhysioNet (44 tentativas de mão direita, 740 s) com limiar
0,80: cerca de **43% das tentativas geram um comando**, com **um disparo fora de hora a cada
53 segundos**. Ou seja: mais da metade das tentativas não fecha a mão, e a mão fecha sozinha
aproximadamente uma vez por minuto.

### Consequências práticas

Um sistema com esse desempenho serve para **demonstração e pesquisa**. Não serve para uso
assistivo sem supervisão. Na prática:

- a órtese precisa estar sempre removível
- o usuário nunca deve ficar sozinho com o sistema em funcionamento
- o curso mecânico deve ser conferido, com a órtese vazia, antes de cada sessão
- rode `testar_ortese.py --real` no início de cada sessão

O firmware em `firmware_esp32/` vai até 180°; quem limita o curso à amplitude da órtese é o
PC (`config.SERVO_AMPLITUDE = 30`). Isso basta enquanto o PC for quem comanda, mas **não
protege** contra um terminal serial aberto por engano nem contra o programa travar com a mão
fechada. Para uso com a órtese vestida, o batente deve estar também no firmware, com retorno
automático ao repouso na ausência de comandos.

---

## Citação

Se este material for útil no seu trabalho, cite o DOI do depósito no Zenodo (ver
[CITATION.cff](CITATION.cff)). Os dados do PhysioNet têm citação própria:

> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a New
> Research Resource for Complex Physiologic Signals. *Circulation*, 101(23), e215–e220.

> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004). BCI2000:
> A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on Biomedical
> Engineering*, 51(6), 1034–1043.
