# -*- coding: utf-8 -*-
"""
verificar.py — confere se este Python tem o que o projeto precisa.

Rode antes de tudo. Se você tem mais de um Python instalado (comum no Windows), é fácil
executar os scripts com o interpretador errado e receber só um `ModuleNotFoundError`
seco, sem pista de qual usar.

    python verificar.py
"""
import importlib.util
import os
import sys

NUCLEO = ["numpy", "scipy", "pandas", "torch", "mne", "matplotlib"]
HARDWARE = {"brainflow": "capacete (capturar.py, tempo_real.py)",
            "serial": "órtese (pyserial)",
            "cv2": "tela de dicas (opencv-python)"}
RAIZ = os.path.dirname(os.path.abspath(__file__))


def falta(mod):
    try:
        return importlib.util.find_spec(mod) is None
    except (ImportError, ValueError):
        return True


def main():
    print(f"Python {sys.version.split()[0]}")
    print(f"  {sys.executable}\n")

    sem_nucleo = [m for m in NUCLEO if falta(m)]
    print("Dependências do treino e da inferência offline:")
    for m in NUCLEO:
        print(f"  {'falta' if m in sem_nucleo else '  ok '}  {m}")

    sem_hw = [m for m in HARDWARE if falta(m)]
    print("\nDependências de hardware (só para capturar e tempo real):")
    for m, para in HARDWARE.items():
        print(f"  {'falta' if m in sem_hw else '  ok '}  {m:<10s} {para}")

    print("\nDados:")
    for nome, caminho, padrao in (
            ("PhysioNet", os.path.join(RAIZ, "dados", "physionet"), "*.edf"),
            ("gravações", os.path.join(RAIZ, "dados", "gravacoes"), "*/eventos.csv"),
            ("modelo", os.path.join(RAIZ, "modelos"), "*.pt")):
        import glob
        n = len(glob.glob(os.path.join(caminho, "**", padrao), recursive=True))
        print(f"  {'  ok ' if n else 'vazio'}  {nome:<10s} {n} arquivo(s) em {os.path.relpath(caminho, RAIZ)}")

    print()
    if sem_nucleo:
        print("Faltam dependências do núcleo. Instale NESTE interpretador:")
        print(f"  \"{sys.executable}\" -m pip install -r requirements.txt")
        print("\nSe você tem vários Pythons instalados, use o caminho completo do que")
        print("tiver as bibliotecas, em vez de confiar no `python` do PATH.")
        return 1
    print("Tudo certo para treinar e inferir offline.")
    if sem_hw:
        print(f"Faltam {', '.join(sem_hw)} — só necessários com o capacete ou a órtese ligados.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
