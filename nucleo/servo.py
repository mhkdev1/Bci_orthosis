# -*- coding: utf-8 -*-
"""
servo.py — comando da órtese, do lado do PC.

Este módulo é a única parte do projeto que fala com o hardware da órtese; tudo que
decide *quando* acionar está fora daqui, o que permite testar a lógica inteira sem nada
plugado (`Ortese(simular=True)`).

------------------------------------------------------------------------------
O PROTOCOLO É O DO FIRMWARE, NÃO O CONTRÁRIO
------------------------------------------------------------------------------
Os valores aqui espelham `firmware_esp32/firmware_esp32.ino`, que é o que está gravado
no ESP32. Ele é INCREMENTAL e trabalha na escala do próprio servo:

    'L'        angle = max(-30, angle - passo)
    'R'        angle = min(180, angle + passo)
    'C'        angle = 90              <- o repouso é 90°, o centro do curso
    '0'..'9'   passo = dígito, com '0' virando 1  (portanto 1 a 9 graus)

Um byte por comando, sem terminador. Depois de cada byte recebido o firmware imprime
`angle=<n>`.

Três consequências que o código abaixo trata explicitamente:

1. O repouso é 90°, não 0°. "Fechar 30°" quer dizer ir de 90 para 120 (ou para 60, se a
   sua montagem fechar no outro sentido). É por isso que `config.SERVO_AMPLITUDE` existe
   separado do ângulo absoluto.

2. O dígito É o passo, não passo-1. Mandar '3' dá passos de 3°. Como o máximo é '9', não
   existe passo de 10°.

3. O firmware informa o ângulo a cada comando. Isso é melhor do que contar às cegas, e a
   leitura é obrigatória de qualquer jeito: se ninguém consumir o eco, ele se acumula no
   buffer do host durante a sessão toda. Aqui o eco é lido e usado como posição real.

------------------------------------------------------------------------------
POR QUE 30 GRAUS DE AMPLITUDE
------------------------------------------------------------------------------
O mecânico é o que manda: a haste da montagem usada força a articulação do dedo acima
disso, e o batente físico não perdoa erro de software.

O segundo motivo é metodológico. O que a rede detecta não é contração muscular: é a
dessincronização sensório-motora (ERD) sobre o córtex, medida em C3/C4/Cz. Esse sinal
indica *intenção* de mover, com latência de ~2 s, e não carrega amplitude — não há como
inferir dele "quanto" fechar. Um arco curto e fixo é o comando honesto para um detector
binário: sinaliza que a intenção foi reconhecida, sem fingir uma proporcionalidade que a
medida não tem.
"""
from __future__ import annotations

import re
import time

import config as cfg

_RE_ANGULO = re.compile(rb"angle\s*=\s*(-?\d+)")


class Ortese:
    """Conexão com a órtese. Use como context manager para ela voltar ao repouso no fim.

        with Ortese() as o:
            o.fechar()
    """

    def __init__(self, porta=None, simular=False, passo_graus=None, sentido=None,
                 verboso=True, detalhe=False):
        self.simular = simular
        self.verboso = verboso
        self.detalhe = detalhe               # True imprime cada byte e cada eco
        self.passo = max(1, min(9, int(passo_graus or cfg.SERVO_PASSO_GRAUS)))
        self.sentido = (sentido or cfg.SERVO_SENTIDO_FECHA).upper()
        if self.sentido not in ("L", "R"):
            raise ValueError("sentido deve ser 'L' ou 'R'")
        self.oposto = "L" if self.sentido == "R" else "R"
        self.repouso_graus = int(cfg.SERVO_REPOUSO)
        self.fechado_graus = int(cfg.ANGULO_FECHADO)
        self.angulo = self.repouso_graus
        self.ser = None

        if simular:
            self._log(f"órtese em SIMULAÇÃO — nada é enviado | repouso {self.repouso_graus}°, "
                      f"fechado {self.fechado_graus}°, passo {self.passo}°")
            self._configurar_passo()
            return

        import serial
        porta = porta or cfg.PORTA_SERVO or self._perguntar_porta()
        self.ser = serial.Serial(porta, cfg.BAUD_SERVO, timeout=0.2, write_timeout=0.6)
        try:                                 # pulso DTR/RTS reinicia o ESP32
            self.ser.setDTR(False); self.ser.setRTS(False); time.sleep(0.2)
            self.ser.setDTR(True); self.ser.setRTS(True)
        except Exception:
            pass
        time.sleep(1.8)                      # espera o boot e a mensagem inicial
        self.ser.reset_input_buffer(); self.ser.reset_output_buffer()
        self._configurar_passo()
        self.repouso()                       # parte de uma posição conhecida
        self._log(f"órtese em {porta} | repouso {self.repouso_graus}°, "
                  f"fechado {self.fechado_graus}° (amplitude {cfg.SERVO_AMPLITUDE}°), "
                  f"passo {self.passo}°, fecha com '{self.sentido}'")

    # ---------------- interface ----------------
    def fechar(self):
        """Fecha a mão até a amplitude configurada."""
        return self.ir_para(self.fechado_graus)

    def abrir(self):
        """Devolve a mão ao repouso."""
        return self.repouso()

    def repouso(self):
        """Comando 'C'. É o único ponto em que a posição é conhecida sem depender da conta."""
        antes = self.angulo
        self._enviar(b"C")
        lido = self._ler_eco()
        self.angulo = lido if lido is not None else self.repouso_graus
        if antes != self.angulo:
            self._log(f"  órtese -> repouso ({self.angulo}°)")
        return self.angulo

    def ir_para(self, graus):
        """Move até o ângulo pedido, em passos, sem passar da amplitude permitida.

        Devolve o ângulo alcançado, que pode não ser exatamente o pedido: o servo só anda
        em múltiplos do passo, e a conta trunca para não ultrapassar o alvo. Com passo de
        3° e amplitude de 30°, os 10 pulsos chegam exatos.
        """
        alvo = self._seguro(graus)
        delta = alvo - self.angulo
        if abs(delta) < self.passo:
            return self.angulo
        letra = self.sentido if (delta > 0) == (self.sentido == "R") else self.oposto
        n = int(abs(delta) // self.passo)
        for _ in range(n):
            self._enviar(letra.encode())
            time.sleep(cfg.SERVO_INTERVALO_S)
        lido = self._ler_eco()
        esperado = self.angulo + (n * self.passo if delta > 0 else -n * self.passo)
        self.angulo = lido if lido is not None else esperado
        if lido is not None and lido != esperado:
            self._log(f"  [aviso] o firmware reportou {lido}° e a conta dava {esperado}°. "
                      f"Seguindo pelo firmware.")
        self._log(f"  órtese -> {self.angulo}° ({n}x '{letra}', passo {self.passo}°)")
        return self.angulo

    def ping(self):
        """True se o firmware respondeu o eco. Manda 'C', que é inofensivo."""
        if self.simular:
            return True
        try:
            self.ser.reset_input_buffer()
            self._enviar(b"C")
            return self._ler_eco() is not None
        except Exception:
            return False

    # ---------------- interno ----------------
    def _configurar_passo(self):
        """Manda o dígito do passo. No firmware o dígito É o passo (só '0' vira 1)."""
        self._enviar(str(self.passo).encode())
        self._ler_eco()

    def _seguro(self, graus):
        """Recorta ao intervalo entre repouso e fechado. Último ponto antes do hardware."""
        alvo = int(round(float(graus)))
        piso, teto = sorted((self.repouso_graus, self.fechado_graus))
        return max(piso, min(teto, alvo))

    def _enviar(self, token):
        if self.detalhe:
            print(f"    {'[sim] ' if self.simular else ''}TX {token.decode()}")
        if self.simular:
            return
        self.ser.write(token)                # byte solto, sem terminador
        self.ser.flush()

    def _ler_eco(self, espera=0.12):
        """Lê o `angle=<n>` que o firmware imprime. Devolve o último, ou None.

        Ler não é opcional: o firmware responde a cada byte, e se ninguém consumir, o
        buffer do host acumula durante a sessão inteira. Aproveitamos para usar o valor
        como posição real, em vez de confiar só na contagem local.
        """
        if self.simular:
            return None
        time.sleep(espera)
        try:
            bruto = self.ser.read_all() or b""
        except Exception:
            return None
        if self.detalhe and bruto.strip():
            print(f"    RX {bruto.strip().decode(errors='ignore')}")
        achados = _RE_ANGULO.findall(bruto)
        return int(achados[-1]) if achados else None

    def _log(self, msg):
        if self.verboso:
            print(msg)

    @staticmethod
    def _perguntar_porta():
        import serial.tools.list_ports
        portas = list(serial.tools.list_ports.comports())
        if not portas:
            raise SystemExit("Nenhuma porta serial encontrada. Confira o cabo do ESP32.")
        for i, p in enumerate(portas):
            print(f"  [{i}] {p.device} — {p.description}")
        escolha = input("Índice da porta (Enter = 0): ").strip()
        return portas[int(escolha) if escolha else 0].device

    def fechar_conexao(self):
        """Devolve a mão ao repouso e desliga a serial. Chamada ao sair do `with`.

        Voltar ao repouso antes de fechar a porta não é cortesia: se o processo terminar
        com a órtese fechada, ela fica fechada no dedo de alguém até o próximo comando.
        """
        try:
            if self.ser is not None:
                self.repouso()
                time.sleep(0.3)
                self.ser.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.fechar_conexao()
        return False


class Decisor:
    """Transforma a decisão ruidosa de cada janela num comando estável para a órtese.

    Três mecanismos, e o segundo é o que mais importa:

    1. LIMIAR — a janela só conta como movimento se a probabilidade passar de
       `config.LIMIAR`. Abaixo disso é tratada como descanso.

    2. TRAVA — depois de fechar, o estado fica travado por `config.SEGURA_S` sem
       reavaliar. Isso não é conveniência, é necessidade: a evidência que a rede usa é o
       transiente da transição, e ele some da janela deslizante poucos segundos depois do
       início do movimento. Exigir confirmação contínua faria a mão reabrir sozinha
       enquanto o usuário ainda está tentando fechá-la.

    3. REFRATÁRIO — depois de soltar, um tempo morto antes de poder acionar de novo,
       para a órtese não ficar batendo.

    A votação por maioria (M de N janelas) foi testada e descartada: medida em 6 runs
    contínuos, ela derruba a detecção junto com o ruído, porque o pico de evidência dura
    cerca de uma janela e os disparos falsos também são isolados. Exigir 3 de 4 zerava as
    duas coisas. O parâmetro continua exposto para quem quiser re-testar com outro modelo.
    """

    def __init__(self, classe=None, limiar=None, segura=None, refratario=None, votos=(1, 1)):
        import collections
        self.classe = classe or cfg.CLASSE_ACIONA
        self.limiar = cfg.LIMIAR if limiar is None else limiar
        self.segura = cfg.SEGURA_S if segura is None else segura
        self.refratario = cfg.REFRATARIO_S if refratario is None else refratario
        self.m, n = votos
        self.hist = collections.deque(maxlen=n)
        self.travado_ate = None
        self.livre_em = 0.0

    def rotulo(self, prob):
        """Rótulo da janela isolada: movimento só acima do limiar, senão descanso."""
        for i, nome in sorted(cfg.CLASSES.items()):
            if nome != "rest" and prob[i] > self.limiar:
                return nome
        return "rest"

    def passo(self, prob, agora):
        """(rótulo da janela, ação) com ação em {None, 'FECHAR', 'ABRIR'}."""
        r = self.rotulo(prob)
        if self.travado_ate is not None:
            if agora >= self.travado_ate:
                self.travado_ate = None
                self.livre_em = agora + self.refratario
                self.hist.clear()
                return r, "ABRIR"
            return r, None
        self.hist.append(r)
        if agora < self.livre_em:
            return r, None
        if self.hist.count(self.classe) >= self.m:
            self.travado_ate = agora + self.segura
            self.hist.clear()
            return r, "FECHAR"
        return r, None
