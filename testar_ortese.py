# -*- coding: utf-8 -*-
"""
testar_ortese.py — testa a comunicação com a órtese, com ou sem hardware.

    python testar_ortese.py              # emula o ESP32 em software, não precisa de nada
    python testar_ortese.py --real       # usa a órtese de verdade, na porta do config

O modo emulado reimplementa em Python, byte a byte, a lógica do
`firmware_esp32/firmware_esp32.ino` — incluindo os limites, o passo e o eco `angle=<n>`.
Se o PC e o firmware discordarem em alguma regra, o teste falha aqui, sem hardware e sem
risco de forçar o dedo de alguém.

O que é verificado:
  1. o dígito do passo é interpretado igual dos dois lados
  2. 'C' leva ao repouso e o PC reconhece o valor
  3. fechar chega exatamente à amplitude configurada
  4. o PC nunca manda o servo além da amplitude, mesmo pedindo mais
  5. a cadeia decisor -> órtese produz a sequência certa de comandos
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

import config as cfg
from nucleo import servo


class ESP32Emulado:
    """Reimplementa o firmware_esp32.ino. Serve de dublê de `serial.Serial`.

    A lógica abaixo é a tradução linha a linha do .ino — se você mudar o firmware, mude
    aqui junto, senão o teste passa a validar uma coisa que não existe mais.
    """
    MIN_ANG, MAX_ANG, CENTRO = -30, 180, 90

    def __init__(self, registrar=False):
        self.angle = self.CENTRO
        self.step = 3
        self.saida = bytearray()
        self.is_open = True
        self.recebidos = []
        self.registrar = registrar

    # --- interface que a classe Ortese usa ---
    def write(self, dados):
        for b in dados:
            self._processar(chr(b))
        return len(dados)

    def flush(self):
        pass

    def read_all(self):
        d, self.saida = bytes(self.saida), bytearray()
        return d

    def reset_input_buffer(self):
        self.saida.clear()

    def reset_output_buffer(self):
        pass

    def close(self):
        self.is_open = False

    def setDTR(self, _):
        pass

    def setRTS(self, _):
        pass

    # --- a lógica do .ino ---
    def _processar(self, c):
        self.recebidos.append(c)
        if c == "L":
            self.angle = max(self.MIN_ANG, self.angle - self.step)
        elif c == "R":
            self.angle = min(self.MAX_ANG, self.angle + self.step)
        elif c == "C":
            self.angle = self.CENTRO
        elif "0" <= c <= "9":
            self.step = int(c) or 1
            self.step = min(10, self.step)
            self.saida += f"Novo passo: {self.step}\n".encode()
        # o firmware imprime o ângulo a cada byte recebido, qualquer que seja
        self.saida += f"angle={self.angle}\n".encode()


def conectar_emulado(**kw):
    """Cria uma Ortese ligada ao ESP32 emulado, sem tocar em serial de verdade."""
    esp = ESP32Emulado()
    o = servo.Ortese.__new__(servo.Ortese)
    o.simular = False
    o.verboso = kw.get("verboso", False)
    o.detalhe = kw.get("detalhe", False)
    o.passo = max(1, min(9, int(kw.get("passo_graus") or cfg.SERVO_PASSO_GRAUS)))
    o.sentido = (kw.get("sentido") or cfg.SERVO_SENTIDO_FECHA).upper()
    o.oposto = "L" if o.sentido == "R" else "R"
    o.repouso_graus = int(cfg.SERVO_REPOUSO)
    o.fechado_graus = int(cfg.ANGULO_FECHADO)
    o.angulo = o.repouso_graus
    o.ser = esp
    o._configurar_passo()
    o.repouso()
    return o, esp


def checar(descricao, obtido, esperado):
    ok = obtido == esperado
    print(f"  {'ok  ' if ok else 'FALHA'} {descricao}")
    if not ok:
        print(f"        esperado {esperado}, obtido {obtido}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="Testa a comunicação com a órtese.")
    ap.add_argument("--real", action="store_true", help="usa o hardware, não o emulador")
    ap.add_argument("--detalhe", action="store_true", help="mostra cada byte trocado")
    a = ap.parse_args()

    print("=" * 74)
    print("  TESTE DA ÓRTESE — " + ("HARDWARE REAL" if a.real else "ESP32 EMULADO"))
    print("=" * 74)
    print(f"  repouso ....: {cfg.SERVO_REPOUSO}°")
    print(f"  amplitude ..: {cfg.SERVO_AMPLITUDE}° -> fechado em {cfg.ANGULO_FECHADO}°")
    print(f"  passo ......: {cfg.SERVO_PASSO_GRAUS}° | fecha com '{cfg.SERVO_SENTIDO_FECHA}'")
    print(f"  pulsos p/ fechar: {cfg.SERVO_AMPLITUDE // cfg.SERVO_PASSO_GRAUS}")

    if a.real:
        print("\n  Mantenha a órtese VAZIA. Ela vai fechar e abrir algumas vezes.")
        o = servo.Ortese(verboso=True, detalhe=a.detalhe)
        esp = None
    else:
        o, esp = conectar_emulado(detalhe=a.detalhe, verboso=a.detalhe)

    tudo_ok = True
    print("\n--- 1. passo e repouso " + "-" * 50)
    if esp:
        tudo_ok &= checar("o firmware entendeu o passo que o PC mandou",
                          esp.step, cfg.SERVO_PASSO_GRAUS)
        tudo_ok &= checar("'C' levou o servo ao centro", esp.angle, cfg.SERVO_REPOUSO)
    tudo_ok &= checar("o PC sabe que está no repouso", o.angulo, cfg.SERVO_REPOUSO)

    print("\n--- 2. fechar " + "-" * 59)
    alcancado = o.fechar()
    tudo_ok &= checar("o PC chegou à amplitude pedida", alcancado, cfg.ANGULO_FECHADO)
    if esp:
        tudo_ok &= checar("o servo está no mesmo ângulo", esp.angle, cfg.ANGULO_FECHADO)
        n_pulsos = sum(1 for c in esp.recebidos if c in ("L", "R"))
        tudo_ok &= checar("número de pulsos enviados",
                          n_pulsos, cfg.SERVO_AMPLITUDE // cfg.SERVO_PASSO_GRAUS)

    print("\n--- 3. o limite segura " + "-" * 51)
    excesso = o.ir_para(cfg.ANGULO_FECHADO + 60)
    tudo_ok &= checar("pedir 60° a mais não move nada", excesso, cfg.ANGULO_FECHADO)
    if esp:
        tudo_ok &= checar("o servo não passou do limite", esp.angle, cfg.ANGULO_FECHADO)

    print("\n--- 4. abrir " + "-" * 60)
    tudo_ok &= checar("voltou ao repouso", o.abrir(), cfg.SERVO_REPOUSO)

    print("\n--- 5. cadeia decisor -> órtese " + "-" * 42)
    dec = servo.Decisor()
    idx = {v: k for k, v in cfg.CLASSES.items()}[dec.classe]
    roteiro = []
    for t, p in [(0.0, [0.9, 0.05, 0.05]), (1.0, None), (2.0, [0.2, 0.1, 0.7]),
                 (6.0, [0.9, 0.05, 0.05])]:
        if p is None:                                   # janela que aciona
            p = [0.05, 0.05, 0.05]; p[idx] = 0.95
        _, acao = dec.passo(np.array(p), t)
        if acao == "FECHAR":
            o.fechar()
        elif acao == "ABRIR":
            o.abrir()
        roteiro.append(acao)
    tudo_ok &= checar("sequência de ações do decisor",
                      roteiro, [None, "FECHAR", None, "ABRIR"])
    tudo_ok &= checar("terminou aberta", o.angulo, cfg.SERVO_REPOUSO)

    if a.real:
        o.fechar_conexao()

    print("\n" + "=" * 74)
    if tudo_ok:
        print("  Tudo certo. O PC e o firmware concordam em todas as regras testadas.")
        if not a.real:
            print("  Isto valida a LÓGICA. Rode com --real, e a órtese vazia, para conferir")
            print("  a fiação, o sentido de fechamento e o batente mecânico.")
    else:
        print("  Há divergências acima. NÃO use com a órtese vestida até resolver.")
    print("=" * 74)
    return 0 if tudo_ok else 1


if __name__ == "__main__":
    sys.exit(main())
