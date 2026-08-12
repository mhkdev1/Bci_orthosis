# -*- coding: utf-8 -*-
"""
treinar.py — treina a rede do zero, ou continua de um checkpoint.

    python treinar.py                          # PhysioNet + gravações, 35 épocas
    python treinar.py --conferir               # só monta o dataset e mostra as contas
    python treinar.py --sem-gravacoes          # só PhysioNet
    python treinar.py --sujeitos 20            # subconjunto, para um teste rápido
    python treinar.py --continuar-de modelos/cnn2d_transicao.pt --so-gravacoes

O checkpoint guarda os pesos, a arquitetura, a estatística de normalização e a geometria
da janela. É o bastante para a inferência reconstruir a mesma cadeia sem depender de
nada do config — o que evita o modo de falha clássico deste tipo de projeto: mudar um
parâmetro meses depois e passar a alimentar o modelo com uma janela diferente da que ele
aprendeu, sem nenhum erro aparecer.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn

import config as cfg
from nucleo import dados, preproc
from nucleo.modelo import CNN2D


def montar_dataset(usar_edf=True, usar_gravacoes=True, limite_sujeitos=None, quieto=False):
    """(X, y, origem). origem: 0 = PhysioNet, 1 = gravação da Cyton."""
    Xs, ys, origens = [], [], []

    if usar_edf:
        idx = dados.sujeitos_physionet()
        sujeitos = dados.bons_sujeitos(list(idx))
        if limite_sujeitos:
            sujeitos = sujeitos[:limite_sujeitos]
        arquivos = [p for s in sujeitos for r, p in idx[s].items() if r in cfg.RUNS_EDF]
        print(f"PhysioNet: {len(arquivos)} arquivos de {len(sujeitos)} sujeitos")
        for i, f in enumerate(arquivos, 1):
            saida = dados.janelas_edf(f)
            if saida is None:
                continue
            X, y = saida[0], saida[1]
            Xs.append(X); ys.append(y); origens.append(np.zeros(len(y), np.int64))
            if not quieto and (i % 20 == 0 or i == len(arquivos)):
                print(f"  ...{i}/{len(arquivos)}")

    if usar_gravacoes:
        n = 0
        for sessao in dados.sessoes_gravadas():
            saida = dados.janelas_gravacao(os.path.join(cfg.DIR_GRAVACOES, sessao))
            if saida is None:
                continue
            X, y, _ = saida
            Xs.append(X); ys.append(y); origens.append(np.ones(len(y), np.int64))
            n += len(y)
            print(f"Gravação {sessao}: {len(y)} janelas")
        if n == 0:
            print("Gravações: nenhuma encontrada em dados/gravacoes/")

    if not Xs:
        sys.exit("Nenhum dado. Confira dados/physionet/ e dados/gravacoes/.")
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(ys).astype(np.int64),
            np.concatenate(origens).astype(np.int64))


def separar(y, fracao, semente):
    """Índices de treino e validação, estratificados por classe.

    Atenção ao interpretar o resultado: o sorteio é por classe, NÃO por sujeito. Janelas
    do mesmo sujeito caem dos dois lados, então a acurácia de validação é otimista para
    o caso "sujeito novo" — que é justamente o caso de quem coloca o capacete pela
    primeira vez. Para medir isso, treine sem um sujeito e infira só nele.
    """
    rng = np.random.default_rng(semente)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = max(1, int(round(len(idx) * fracao)))
        va.extend(idx[:n]); tr.extend(idx[n:])
    rng.shuffle(tr); rng.shuffle(va)
    return np.array(tr), np.array(va)


def aumentar(x):
    """Ruído, escala e deslocamento — aplicados só no treino, sorteados por janela."""
    if random.random() < cfg.AUG_PROB:
        x = x * random.uniform(*cfg.AUG_ESCALA)
    if cfg.AUG_DESLOCAMENTO > 0 and random.random() < cfg.AUG_PROB:
        x = torch.roll(x, shifts=random.randint(-cfg.AUG_DESLOCAMENTO, cfg.AUG_DESLOCAMENTO), dims=-1)
    if cfg.AUG_RUIDO > 0 and random.random() < cfg.AUG_PROB:
        x = x + torch.randn_like(x) * cfg.AUG_RUIDO
    return x


class Conjunto(torch.utils.data.Dataset):
    def __init__(self, X, y, treino):
        self.X, self.y, self.treino = torch.from_numpy(X), torch.from_numpy(y), treino

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]
        return (aumentar(x.clone()) if (self.treino and cfg.AUGMENT) else x), self.y[i]


def em_lotes(modelo, X, lote=None):
    """Logits em lotes. Um forward único estoura a RAM: o banco temporal expande cada
    janela para 100 mapas, e 3.700 janelas de 526 pontos passam de 2 GB só nessa saída."""
    lote = lote or cfg.LOTE_VALIDACAO
    modelo.eval()
    with torch.no_grad():
        return torch.cat([modelo(X[i:i + lote]) for i in range(0, len(X), lote)])


def matriz_confusao(y, pred):
    n = len(cfg.CLASSES)
    m = np.zeros((n, n), int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m


def relatorio(m):
    """Imprime precisão, recall e F1 por classe. Devolve (acurácia, macro-F1)."""
    nomes = [cfg.CLASSES[i] for i in sorted(cfg.CLASSES)]
    print(f"\n  {'classe':>8s} | {'precisão':>9s} {'recall':>8s} {'F1':>8s} | {'n':>5s}")
    print("  " + "-" * 49)
    f1s = []
    for i, nome in enumerate(nomes):
        p = m[i, i] / m[:, i].sum() if m[:, i].sum() else 0.0
        r = m[i, i] / m[i].sum() if m[i].sum() else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1)
        print(f"  {nome:>8s} | {p*100:8.1f}% {r*100:7.1f}% {f1:8.3f} | {m[i].sum():5d}")
    print("  " + "-" * 49)
    acc = np.trace(m) / m.sum()
    print(f"  {'acurácia':>8s} | {'':>9s} {'':>8s} {acc:8.3f} | {m.sum():5d}")
    print(f"  {'macro-F1':>8s} | {'':>9s} {'':>8s} {np.mean(f1s):8.3f} |")
    mov = m[1:]
    if mov.sum():
        print(f"\n  Movimento classificado como descanso: {mov[:, 0].sum()}/{mov.sum()} "
              f"({mov[:, 0].sum()/mov.sum()*100:.1f}%) — é o erro que impede a órtese de fechar.")
    return acc, float(np.mean(f1s))


def main():
    ap = argparse.ArgumentParser(description="Treina a CNN de motor imagery.")
    ap.add_argument("--saida", default=os.path.join(cfg.DIR_MODELOS, "cnn2d_novo.pt"))
    ap.add_argument("--epocas", type=int, default=cfg.EPOCAS)
    ap.add_argument("--lote", type=int, default=cfg.LOTE)
    ap.add_argument("--lr", type=float, default=cfg.TAXA_APRENDIZADO)
    ap.add_argument("--sujeitos", type=int, default=None, help="usa só os N primeiros")
    ap.add_argument("--sem-gravacoes", action="store_true", help="só PhysioNet")
    ap.add_argument("--so-gravacoes", action="store_true", help="só as gravações da Cyton")
    ap.add_argument("--continuar-de", default=None, metavar="CKPT",
                    help="parte dos pesos deste checkpoint (o original não é alterado)")
    ap.add_argument("--conferir", action="store_true", help="monta o dataset e para")
    a = ap.parse_args()

    torch.manual_seed(cfg.SEMENTE); random.seed(cfg.SEMENTE); np.random.seed(cfg.SEMENTE)

    print("=" * 74)
    print(f"  Janela: {cfg.AMOSTRAS} amostras = {cfg.JANELA_S:.2f} s @ {cfg.TAXA_IA} Hz, "
          f"centrada na transição")
    print(f"  Canais: {cfg.CANAIS} | classes: {list(cfg.CLASSES.values())}")
    print("=" * 74)

    X, y, origem = montar_dataset(usar_edf=not a.so_gravacoes,
                                  usar_gravacoes=not a.sem_gravacoes,
                                  limite_sujeitos=a.sujeitos)
    contagem = np.bincount(y, minlength=len(cfg.CLASSES))
    print(f"\nDataset: {len(y)} janelas | "
          f"{ {cfg.CLASSES[i]: int(contagem[i]) for i in range(len(contagem))} }")
    print(f"  PhysioNet: {int((origem == 0).sum())} | gravações: {int((origem == 1).sum())}")
    if a.conferir:
        print(f"  X{X.shape}\nOK — rode sem --conferir para treinar.")
        return

    tr, va = separar(origem * 10 + y, cfg.FRACAO_VALIDACAO, cfg.SEMENTE)
    mu, sd = preproc.estatistica(X[tr])       # estatística SÓ do treino
    X = preproc.normalizar(X, mu, sd)

    modelo = CNN2D(len(cfg.CANAIS), len(cfg.CLASSES), cfg.AMOSTRAS,
                   n_por_escala=cfg.CNN_FILTROS, kernels=cfg.CNN_KERNELS,
                   depth=cfg.CNN_DEPTH, n_saida=cfg.CNN_SAIDA)
    n_par = sum(p.numel() for p in modelo.parameters())
    print(f"\nRede: {len(cfg.CNN_KERNELS)} escalas x {cfg.CNN_FILTROS} filtros | {n_par:,} parâmetros")

    if a.continuar_de:
        ck = torch.load(a.continuar_de, map_location="cpu", weights_only=False)
        modelo.load_state_dict(ck["state_dict"])
        print(f"Continuando de {os.path.basename(a.continuar_de)} — o original não é alterado")

    peso = None
    if cfg.PESO_POR_CLASSE:
        w = contagem.sum() / np.clip(contagem, 1, None)
        peso = torch.tensor(w / w.sum() * len(w), dtype=torch.float32)
    perda = nn.CrossEntropyLoss(weight=peso)
    otim = torch.optim.Adam(modelo.parameters(), lr=a.lr)

    carga = torch.utils.data.DataLoader(Conjunto(X[tr], y[tr], True),
                                        batch_size=a.lote, shuffle=True)
    Xv, yv = torch.from_numpy(X[va]), torch.from_numpy(y[va])

    print(f"Treinando {a.epocas} épocas na CPU | treino={len(tr)} validação={len(va)}")
    print("-" * 74)
    melhor, melhor_ep, melhor_pesos = 0.0, 0, None
    for ep in range(1, a.epocas + 1):
        t0 = time.time()
        modelo.train()
        soma, acertos, n = 0.0, 0, 0
        for xb, yb in carga:
            otim.zero_grad()
            logits = modelo(xb)
            L = perda(logits, yb)
            L.backward(); otim.step()
            soma += L.item() * len(xb)
            acertos += int((logits.argmax(1) == yb).sum()); n += len(xb)
        lv = em_lotes(modelo, Xv)
        with torch.no_grad():
            vperda = float(perda(lv, yv))
            vacc = float((lv.argmax(1) == yv).float().mean())
        print(f"  época {ep:3d}/{a.epocas} | loss {soma/n:.4f} | treino {acertos/n*100:5.1f}% "
              f"| val_loss {vperda:.4f} | validação {vacc*100:5.1f}% | {time.time()-t0:4.0f}s")
        if vacc >= melhor:
            melhor, melhor_ep = vacc, ep
            melhor_pesos = {k: v.clone() for k, v in modelo.state_dict().items()}

    modelo.load_state_dict(melhor_pesos); modelo.eval()
    pred = em_lotes(modelo, Xv).argmax(1).numpy()
    print("-" * 74)
    print(f"Melhor validação: {melhor*100:.1f}% (época {melhor_ep})")
    relatorio(matriz_confusao(y[va], pred))

    esquema = {"nome": "centrado-transicao", "meia_janela_s": cfg.MEIA_JANELA,
               "taxa": cfg.TAXA_IA, "canais": list(cfg.CANAIS),
               "treinado_em": datetime.now().isoformat(timespec="seconds")}
    os.makedirs(os.path.dirname(a.saida) or ".", exist_ok=True)
    torch.save({"state_dict": modelo.state_dict(), "arch": modelo.arch,
                "n_canais": modelo.n_canais, "n_classes": modelo.n_classes,
                "amostras": modelo.amostras, "norm_mu": mu, "norm_sd": sd,
                "esquema": esquema}, a.saida)
    print(f"\nModelo salvo em {a.saida}")


if __name__ == "__main__":
    main()
