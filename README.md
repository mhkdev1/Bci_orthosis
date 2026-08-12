# BCI de imagética motora para acionamento de órtese

Interface cérebro-computador que lê EEG de três eletrodos, classifica a intenção de
movimento em **descanso / mão esquerda / mão direita**, e usa essa decisão para fechar
uma órtese de mão acionada por servo.

O pacote traz o código completo, os dados para reproduzir os resultados e um modelo já
treinado. Dá para rodar as três coisas — treino, inferência offline e inferência ao vivo
com a órtese — sem baixar mais nada, tirando as bibliotecas Python.

---

## O que tem aqui

```
bci_ortese/
├── config.py               todos os parâmetros, num arquivo só
├── treinar.py              treina a rede
├── inferir_offline.py      roda o modelo em dados gravados e mede a acurácia
├── tempo_real.py           capacete ligado, órtese acionada ao vivo
├── capturar.py             grava uma sessão nova
├── controle_servo_1.py     controle manual da órtese pelo teclado
├── firmware_esp32/
│   └── firmware_esp32.ino  o que roda DENTRO do ESP32 (servo no GPIO 18)
├── nucleo/
│   ├── preproc.py          filtros, janelamento e normalização
│   ├── modelo.py           a rede convolucional
│   ├── dados.py            leitura do PhysioNet e das gravações
│   └── servo.py            comando da órtese e lógica de acionamento
├── dados/
│   ├── physionet/          10 sujeitos do eegmmidb (6 runs cada)
│   └── gravacoes/          2 sessões gravadas com a OpenBCI Cyton
├── modelos/
│   └── cnn2d_transicao.pt  modelo treinado, pronto para uso
├── figuras/                curvas de treino
├── docs/                   logs de treino originais
└── RELATORIO.md            o método descrito passo a passo
```

O [RELATORIO.md](RELATORIO.md) explica como cada etapa funciona e por quê. Este arquivo
é só o manual de uso.

---

## Instalação

Python 3.9 ou mais novo.

```bash
pip install -r requirements.txt
python verificar.py          # confere se este Python tem tudo
```

Comece pelo `verificar.py`. Ele lista o que falta, mostra qual interpretador está sendo
usado e monta a linha de instalação correta.

**Se você tem mais de um Python instalado** — situação comum no Windows, onde `python` no
PATH pode não ser o mesmo que a sua IDE usa —, instale e execute sempre com o caminho
completo do mesmo interpretador:

```powershell
& "C:\caminho\para\python.exe" -m pip install -r requirements.txt
& "C:\caminho\para\python.exe" inferir_offline.py --sujeito S001
```

O sintoma de estar no interpretador errado é um `ModuleNotFoundError: No module named
'torch'` logo na primeira linha, mesmo tendo instalado tudo.

Para rodar apenas o treino e a inferência offline, bastam `numpy`, `scipy`, `torch`,
`mne`, `pandas` e `matplotlib`. As demais só são necessárias com hardware:
`brainflow` (capacete), `pyserial` (órtese) e `opencv-python` (tela de dicas).

---

## Uso rápido

### 1. Inferência offline — não precisa de hardware

Roda o modelo em dados já gravados e mostra acurácia, F1 por classe e matriz de confusão.

```bash
python inferir_offline.py --listar                    # o que existe em disco
python inferir_offline.py --sujeito S001              # um sujeito do PhysioNet
python inferir_offline.py --sujeito S001 --run 4      # um run só
python inferir_offline.py --gravacao S005_R1_cortado  # uma sessão da Cyton
python inferir_offline.py --gravacao S004_R1 --detalhe
```

Nas gravações contínuas ele faz uma segunda leitura, varrendo o sinal com janela
deslizante — é a simulação do que a órtese enfrentaria em tempo real. Para ver os
comandos que seriam enviados, sem hardware nenhum:

```bash
python inferir_offline.py --gravacao S005_R1_cortado --ortese --simular-ortese
```

Com a órtese ligada, tire o `--simular-ortese`.

### 2. Treino

```bash
python treinar.py --conferir            # monta o dataset e mostra as contas, sem treinar
python treinar.py                       # PhysioNet + gravações, 35 épocas
python treinar.py --sujeitos 5          # subconjunto, para um teste rápido
python treinar.py --sem-gravacoes       # só PhysioNet
```

Roda na CPU. Com os 10 sujeitos deste pacote, leva alguns minutos; o modelo distribuído
foi treinado com 105 sujeitos e levou cerca de duas horas.

### 3. Tempo real com a órtese

```bash
python tempo_real.py --sem-ortese       # só imprime, para testar o capacete sozinho
python tempo_real.py --simular-ortese   # imprime os comandos que mandaria
python tempo_real.py                    # capacete + órtese
```

Antes, ajuste as portas em `config.py`:

```python
PORTA_EEG   = "COM9"     # dongle da Cyton      (Linux: /dev/ttyUSB0)
PORTA_SERVO = "COM3"     # ESP32 da órtese
```

Para descobrir quais são:

```bash
python -c "import serial.tools.list_ports as p; [print(x.device,'-',x.description) for x in p.comports()]"
```

### 4. Gravar uma sessão nova

```bash
python capturar.py --ensaio                  # sem capacete: confere tempos e formato
python capturar.py --sujeito S006 --run R1   # ~4 minutos
```

A gravação vai para `dados/gravacoes/S006_R1/` e já pode ser usada no treino e na
inferência offline.

---

## Montagem dos eletrodos

Três canais no sistema 10-20, sobre o córtex motor:

| canal | posição | o que capta |
|---|---|---|
| C3 | hemisfério esquerdo | movimento da mão **direita** |
| C4 | hemisfério direito | movimento da mão **esquerda** |
| Cz | linha média | referência de atividade motora central |

A ordem em `config.CANAIS` precisa bater com a ordem física na Cyton — é a mesma ordem
que a rede viu no treino, e trocá-la degrada o resultado sem gerar erro nenhum.

---

## A órtese

O ESP32 recebe **um byte por comando, sem terminador**, e o movimento é **incremental**:

| byte | efeito |
|---|---|
| `R` | `angle = min(180, angle + passo)` |
| `L` | `angle = max(-30, angle - passo)` |
| `C` | `angle = 90` — o repouso é o **centro** do curso, não zero |
| `0`–`9` | tamanho do passo, em graus (o dígito *é* o passo; `0` vira 1) |

A cada byte o firmware imprime `angle=<n>`. O lado do PC lê esse eco e o usa como
posição real, em vez de contar pulsos às cegas — e precisa lê-lo de qualquer forma,
senão a saída se acumula no buffer do host durante a sessão inteira.

Fechar 30° quer dizer ir de 90° para 120° (ou para 60°, se a sua montagem fechar no
outro sentido). Com passo de 3°, são 10 pulsos.

Há dois lados, e eles não se confundem:

| arquivo | onde roda | papel |
|---|---|---|
| `firmware_esp32/firmware_esp32.ino` | **dentro do ESP32** | recebe os bytes e move o servo |
| `firmware_esp32_seguro/` | dentro do ESP32 | variante com batente e retorno automático |
| `controle_servo_1.py` | no PC | controle manual por teclado |
| `nucleo/servo.py` | no PC | é quem a inferência usa |

### Gravando o firmware

Abra `firmware_esp32/firmware_esp32.ino` na IDE do Arduino, instale a biblioteca
**ESP32Servo** pelo Gerenciador de Bibliotecas, selecione a sua placa ESP32 e grave.

Ligação:

```
servo (sinal)  ->  GPIO 18
servo (VCC)    ->  fonte externa 5 V   (não use o 5V do ESP32)
servo (GND)    ->  GND comum com o ESP32
```

O GND precisa ser comum, senão o servo treme ou não responde. E alimentar o servo pelo
regulador da placa costuma derrubá-la no pico de corrente da partida.

O firmware em uso vai até 180° — quem limita o curso à amplitude da órtese é o PC
(`config.SERVO_AMPLITUDE = 30`). Isso basta enquanto o PC for quem comanda, mas não
protege contra um terminal aberto por engano nem contra o programa travar com a mão
fechada.

Para uso com a órtese **vestida**, grave a variante `firmware_esp32_seguro/`: ela põe o
batente também no firmware e devolve a órtese ao repouso sozinha se nenhum comando
chegar por 10 s.

### Conferindo a ligação

Primeiro o teste de comunicação, que roda **sem hardware nenhum** — ele emula o ESP32 em
software e confere se o PC e o firmware concordam em todas as regras:

```bash
python testar_ortese.py            # emulado
python testar_ortese.py --real     # com a órtese ligada, e VAZIA
```

Depois o controle manual, que é como você descobre qual sentido fecha a sua montagem:

```bash
python controle_servo_1.py       # setas ← → movem, ↑ volta ao repouso, ESC sai
```

Depois ajuste em `config.py`:

```python
SERVO_SENTIDO_FECHA = "R"    # qual letra fecha a SUA montagem
SERVO_PASSO_GRAUS   = 10     # 1 a 10 graus por pulso
ANGULO_FECHADO      = 30     # curso máximo
```

**O curso é limitado a 30°.** É o limite mecânico confortável da montagem usada: acima
disso a haste força a articulação do dedo. O módulo nunca envia pulsos além do teto, e
trunca para baixo quando o ângulo não é múltiplo do passo — pedir 30° com passo de 4°
para em 28°, nunca em 32°.

Como o firmware é incremental e não devolve a posição, a contagem de ângulo é mantida no
PC e ressincronizada a cada `C`. Toda sequência de fechamento parte do repouso, e o
programa devolve a órtese ao repouso ao terminar, inclusive se for interrompido. Sem
isso, um pulso perdido faria o erro acumular e o limite de 30° deixaria de valer.

## Antes de usar em alguém

Este sistema é um protótipo de pesquisa. Com o limiar padrão, cerca de 43% das tentativas
geram um comando e a órtese dispara sozinha aproximadamente uma vez por minuto. Os
números completos, e como eles foram medidos, estão no [RELATORIO.md](RELATORIO.md).

Na prática isso significa: a órtese precisa estar sempre removível, o usuário nunca deve
ficar sozinho com o sistema em funcionamento, e o curso mecânico deve ser conferido antes
de cada sessão.

---

## Dados incluídos

**PhysioNet — EEG Motor Movement/Imagery Dataset (eegmmidb).** Dez sujeitos, seis runs
cada: 3, 7 e 11 (movimento real da mão) e 4, 8 e 12 (imagética). Base pública, disponível
em <https://physionet.org/content/eegmmidb/1.0.0/>. O modelo distribuído foi treinado com
105 sujeitos; os dez aqui bastam para reproduzir o pipeline e rodar um treino de
demonstração.

**Gravações próprias.** Duas sessões com a OpenBCI Cyton, três eletrodos, 250 Hz:

| sessão | duração | segmentos | observação |
|---|---|---|---|
| `S004_R1` | 248 s | 60 | sessão completa |
| `S005_R1_cortado` | 193 s | 47 | a bateria acabou no meio; a parte instável foi removida |

O corte do S005 não foi arbitrário: a partir do evento 48 os segmentos passaram a mapear
para a mesma amostra, sinal de que a placa parou de transmitir. Tudo até ali está íntegro.

---

## Reprodutibilidade

Os resultados abaixo saem direto dos comandos indicados, sem nenhum ajuste:

| comando | acurácia | macro-F1 |
|---|---|---|
| `python inferir_offline.py --sujeito S001` | 73,6% | 0,723 |
| `python inferir_offline.py --gravacao S005_R1_cortado` | 66,7% | 0,634 |

O treino usa semente fixa (`config.SEMENTE`), mas o resultado exato ainda pode variar um
pouco entre máquinas por causa de diferenças de BLAS e de versão do PyTorch.

---

## Citação

Se este material for útil no seu trabalho, cite o DOI do depósito no Zenodo. Os dados do
PhysioNet têm citação própria:

> Goldberger, A., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of
> a New Research Resource for Complex Physiologic Signals. *Circulation*, 101(23),
> e215–e220.

> Schalk, G., McFarland, D.J., Hinterberger, T., Birbaumer, N., Wolpaw, J.R. (2004).
> BCI2000: A General-Purpose Brain-Computer Interface (BCI) System. *IEEE Transactions on
> Biomedical Engineering*, 51(6), 1034–1043.
