# -*- coding: utf-8 -*-
"""
modelo.py — a rede convolucional usada para classificar as janelas de EEG.

A arquitetura reproduz, em camadas de convolução, o que o FBCSP faz à mão:
filtra em várias bandas, projeta no espaço dos eletrodos, mede a potência e
classifica pelo logaritmo dela. Cada bloco abaixo corresponde a uma dessas etapas.

Entrada  : (lote, 3 canais, T amostras)   — C3, C4, Cz a 128 Hz
Saída    : (lote, 3 classes)              — rest, left, right
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Square(nn.Module):
    """x² — converte amplitude em potência, que é a grandeza do ERD/ERS."""

    def forward(self, x):
        return x * x


class SafeLog(nn.Module):
    """log com piso. Sem o clamp, potência perto de zero vira -inf e mata o gradiente."""

    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        return torch.log(torch.clamp(x, min=self.eps))


class BancoTemporal(nn.Module):
    """Filtros temporais em várias escalas, em paralelo, sobre cada canal.

    Um kernel enxerga um horizonte só: 15 amostras não comportam um ciclo de 3 Hz, e
    65 amostras borram um transiente de beta. Cada ramo cobre uma escala e a saída é a
    concatenação de todos.

    Atenção ao ler as escalas: o comprimento do kernel limita a frequência mais BAIXA
    que ele representa (um ciclo inteiro precisa caber), não a mais alta. A 128 Hz, o
    ciclo cabe a partir de 128/k Hz:

         5 amostras =  39 ms  ->  >= 25,6 Hz   beta alto, até o teto de 35 Hz
        11 amostras =  86 ms  ->  >= 11,6 Hz   beta
        15 amostras = 117 ms  ->  >=  8,5 Hz   mu / alfa
        33 amostras = 258 ms  ->  >=  3,9 Hz   theta
        65 amostras = 508 ms  ->  >=  2,0 Hz   delta

    As escalas cobrem a mesma faixa do passa-banda (0,5-35 Hz). Kernel mais longo que o
    necessário só gasta parâmetro; mais curto deixa a banda baixa sem representação.

    Só kernels ímpares: com k par, o padding k//2 devolve L+1 e os ramos saem com
    comprimentos diferentes, quebrando a concatenação.
    """

    def __init__(self, n_por_escala=20, kernels=(5, 11, 15, 33, 65)):
        super().__init__()
        pares = [k for k in kernels if k % 2 == 0]
        if pares:
            raise ValueError(f"kernels devem ser ímpares (padding 'same'); pares: {pares}")
        self.ramos = nn.ModuleList([
            nn.Conv2d(1, n_por_escala, (1, k), padding=(0, k // 2), bias=False)
            for k in kernels
        ])
        self.out_channels = n_por_escala * len(kernels)

    def forward(self, x):
        return torch.cat([r(x) for r in self.ramos], dim=1)


class CNN2D(nn.Module):
    """Rede completa. ~21 mil parâmetros — pequena de propósito, porque são 3 eletrodos.

    O campo `arch` é gravado no checkpoint. Sem ele, um .pt treinado com outra
    configuração não recarrega (erro de shape) e a inferência quebra.
    """

    def __init__(self, n_canais, n_classes, amostras,
                 n_por_escala=20, kernels=(5, 11, 15, 33, 65), depth=3, n_saida=32):
        super().__init__()
        self.amostras = int(amostras)
        self.n_canais = int(n_canais)
        self.n_classes = int(n_classes)
        self.arch = {"n_por_escala": n_por_escala, "kernels": tuple(kernels),
                     "depth": depth, "n_saida": n_saida}
        F = n_por_escala * len(kernels)          # mapas temporais
        Fs = F * depth                           # depois do filtro espacial

        self.bloco = nn.Sequential(
            # (1) banco temporal multi-escala
            BancoTemporal(n_por_escala, kernels),
            nn.BatchNorm2d(F),
            # (2) filtro espacial depthwise: mistura C3/C4/Cz e aprende a topografia de
            # cada banda. É onde mora a lateralização do ERD. Sem ReLU antes do Square —
            # a não-linearidade da cadeia é o próprio quadrado.
            nn.Conv2d(F, Fs, (n_canais, 1), groups=F, bias=False),
            nn.BatchNorm2d(Fs),
            # (3) potência de banda: square -> média móvel -> log
            Square(),
            nn.AvgPool2d((1, 4)),
            SafeLog(),
            nn.Dropout(0.25),
            # (4) separável: refina o tempo de cada mapa isolado, depois mistura os mapas
            nn.Conv2d(Fs, Fs, (1, 15), padding=(0, 7), groups=Fs, bias=False),
            nn.Conv2d(Fs, n_saida, (1, 1), bias=False),
            nn.BatchNorm2d(n_saida), nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, max(1, amostras // 16)))
        with torch.no_grad():
            saida = self.pool(self.bloco(torch.zeros(2, n_canais, amostras).unsqueeze(1)))
            n_flat = saida.shape[1] * saida.shape[2] * saida.shape[3]
        self.cabeca = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_flat, n_classes))

    def embedding(self, x):
        """Representação antes da camada de classificação."""
        return torch.flatten(self.pool(self.bloco(x.unsqueeze(1))), 1)

    def forward(self, x):
        return self.cabeca(self.embedding(x))


def salvar(caminho, modelo, mu, sd, esquema):
    """Grava pesos + tudo que a inferência precisa para reconstruir a mesma cadeia."""
    import os
    os.makedirs(os.path.dirname(caminho) or ".", exist_ok=True)
    torch.save({
        "state_dict": modelo.state_dict(),
        "arch": modelo.arch,
        "n_canais": modelo.n_canais,
        "n_classes": modelo.n_classes,
        "amostras": modelo.amostras,
        "norm_mu": mu, "norm_sd": sd,
        "esquema": esquema,
    }, caminho)


def carregar(caminho):
    """Devolve (modelo em eval, amostras, esquema). O modelo carrega a normalização junto.

    A estatística de normalização vem do checkpoint, não do dado de entrada: é a mesma
    escala do treino. Sem isso, um sinal com amplitude diferente entra deslocado e o
    modelo responde a escala em vez de responder a padrão.
    """
    ck = torch.load(caminho, map_location="cpu", weights_only=False)
    amostras = int(ck["amostras"])
    m = CNN2D(int(ck["n_canais"]), int(ck["n_classes"]), amostras, **ck["arch"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    m.norm = (ck.get("norm_mu"), ck.get("norm_sd"))
    return m, amostras, ck.get("esquema", {})
