# -*- coding: utf-8 -*-
"""
inferir_offline.py — roda o modelo em dados JÁ GRAVADOS e mede a acurácia.

Funciona com as duas origens, sem diferença para quem usa:

    python inferir_offline.py --listar                 # o que existe em disco
    python inferir_offline.py --sujeito S001           # um sujeito do PhysioNet
    python inferir_offline.py --sujeito S001 --run 4   # um run só
    python inferir_offline.py --gravacao S005_R1_cortado
    python inferir_offline.py --sujeito S001 --ortese  # aciona a órtese de verdade
    python inferir_offline.py --gravacao S004_R1 --ortese --simular-ortese

Duas leituras da mesma inferência, e elas respondem perguntas diferentes:

  POR JANELA — recorta na mesma geometria do treino e classifica. É o número comparável
  à validação do treino, e o que se reporta como "acurácia do modelo".

  JANELA DESLIZANTE — varre a gravação inteira sem saber onde estão os eventos, como o
  tempo real faz, e mede quantos comandos a órtese receberia e quando. Só existe para as
  gravações contínuas, porque exige sinal sem cortes.

A segunda costuma ser bem pior que a primeira. A diferença entre as duas é o custo de
não ter um cue para se alinhar — e é ela que decide se o sistema serve na prática.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch

import config as cfg
from nucleo import dados, preproc, servo
from nucleo.modelo import carregar


def prever(modelo, X):
    """(pred, prob) para um lote (n, C, T) já recortado."""
    Xn = preproc.normalizar_com_modelo(modelo, X)
    with torch.no_grad():
        logits = torch.cat([modelo(torch.from_numpy(Xn[i:i + cfg.LOTE_VALIDACAO]).float())
                            for i in range(0, len(Xn), cfg.LOTE_VALIDACAO)])
        prob = torch.softmax(logits, dim=1).numpy()
    return prob.argmax(1), prob


def secao(titulo):
    print("\n" + "=" * 74)
    print(f"  {titulo}")
    print("=" * 74)


def tabela_metricas(m):
    """Precisão, recall e F1 por classe. Devolve (acurácia, macro-F1, base)."""
    nomes = [cfg.CLASSES[i] for i in sorted(cfg.CLASSES)]
    print(f"  {'classe':>8s} | {'precisão':>9s} {'recall':>8s} {'F1':>8s} | {'n':>5s}")
    print("  " + "-" * 49)
    f1s, sup = [], []
    for i, nome in enumerate(nomes):
        p = m[i, i] / m[:, i].sum() if m[:, i].sum() else 0.0
        r = m[i, i] / m[i].sum() if m[i].sum() else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        f1s.append(f1); sup.append(m[i].sum())
        print(f"  {nome:>8s} | {p*100:8.1f}% {r*100:7.1f}% {f1:8.3f} | {m[i].sum():5d}")
    print("  " + "-" * 49)
    N = m.sum()
    macro = float(np.mean(f1s))
    print(f"  {'macro-F1':>8s} | {'':>9s} {'':>8s} {macro:8.3f} | {N:5d}")
    imaj = int(np.argmax(sup))
    p_maj = sup[imaj] / N
    base = (2 * p_maj / (p_maj + 1.0)) / len(nomes)
    return float(np.trace(m) / N), macro, base, nomes[imaj], p_maj


def confusao(y, pred):
    n = len(cfg.CLASSES)
    m = np.zeros((n, n), int)
    for a, b in zip(y, pred):
        m[a, b] += 1
    return m


def imprimir_confusao(m):
    nomes = [cfg.CLASSES[i] for i in sorted(cfg.CLASSES)]
    titulo = "real / predito"
    print(f"  {titulo:>16s} " + "".join(f"{n:>8s}" for n in nomes))
    for i, nome in enumerate(nomes):
        print(f"  {nome:>16s} " + "".join(f"{v:8d}" for v in m[i]))


def varredura(modelo, sinal, ev, passo, ortese=None, ritmo=False):
    """Janela deslizante sobre o sinal contínuo, exatamente como o tempo real faz.

    `ritmo=True` faz a varredura andar no tempo do SINAL, não na velocidade da CPU. Sem
    isso a varredura corre várias vezes mais rápido que a gravação, e a órtese recebe os
    comandos amontoados — inútil para assistir e ruim para o servo, que não acompanha.
    """
    import time as _t
    taxa = float(cfg.TAXA_PLACA)
    largura = int(np.ceil(cfg.AMOSTRAS / cfg.TAXA_IA * taxa))
    dec = servo.Decisor()
    alvo_cod = [k for k, v in cfg.CODIGO_CLASSE.items()
                if v == {n: i for i, n in cfg.CLASSES.items()}[dec.classe]]
    alvos = [int(r["amostra_inicio"]) / taxa for _, r in ev.iterrows()
             if str(r["code"]) in alvo_cod]

    disparos, i = [], largura
    t0_parede = _t.monotonic()
    while i <= len(sinal):
        x = preproc.janela_bruta(sinal[i - largura:i].T, taxa)
        _, prob = prever(modelo, x[None])
        t_sinal = i / taxa
        _, acao = dec.passo(prob[0], t_sinal)
        if acao == "FECHAR":
            disparos.append(t_sinal)
            if ortese is not None:
                ortese.fechar()
        elif acao == "ABRIR" and ortese is not None:
            ortese.abrir()
        if ritmo:                                  # anda no tempo da gravação
            atraso = (t_sinal - largura / taxa) - (_t.monotonic() - t0_parede)
            if atraso > 0:
                _t.sleep(atraso)
        i += int(round(passo * taxa))

    TOL = 5.0
    lat = [min(x for x in disparos if o <= x <= o + TOL) - o
           for o in alvos if any(o <= x <= o + TOL for x in disparos)]
    falsos = [x for x in disparos if all(not (o <= x <= o + TOL) for o in alvos)]
    dur = len(sinal) / taxa

    print(f"  passo {passo:g} s | limiar {dec.limiar:g} | classe '{dec.classe}' | {dur:.0f} s")
    print(f"  comandos de fechar enviados ..: {len(disparos)}")
    print(f"  cues de '{dec.classe}' detectados : {len(lat)}/{len(alvos)}"
          + (f" | latência mediana {np.median(lat):.2f} s" if lat else ""))
    print(f"  disparos fora de cue .........: {len(falsos)}"
          + (f" (1 a cada {dur/len(falsos):.0f} s)" if falsos else ""))
    if lat:
        print(f"\n  A latência fica perto de {cfg.MEIA_JANELA:.1f} s porque é aí que a janela")
        print("  deslizante reproduz a geometria de treino: a transição no centro.")


def main():
    ap = argparse.ArgumentParser(description="Inferência em dados gravados.")
    ap.add_argument("--modelo", default=cfg.MODELO_PADRAO)
    ap.add_argument("--sujeito", default=None, help="sujeito do PhysioNet, ex.: S001")
    ap.add_argument("--run", type=int, default=None, help="um run só (3,4,7,8,11,12)")
    ap.add_argument("--gravacao", default=None, help="sessão da Cyton, ex.: S005_R1_cortado")
    ap.add_argument("--listar", action="store_true", help="mostra o que existe em disco")
    ap.add_argument("--detalhe", action="store_true", help="imprime janela a janela")
    ap.add_argument("--trials", action="store_true",
                    help="lista os trials disponíveis (classe, índice dentro da classe, instante)")
    ap.add_argument("--trial", nargs=2, metavar=("CLASSE", "N"), default=None,
                    help="roda UM trial só: classe (rest/left/right) e o índice dentro dela, "
                         "contando de 1. Ex.: --trial right 5")
    ap.add_argument("--passo", type=float, default=cfg.PASSO_S)
    ap.add_argument("--ortese", action="store_true", help="aciona a órtese durante a varredura")
    ap.add_argument("--ritmo", action="store_true",
                    help="com --ortese: percorre a gravação no tempo real dela, para dar "
                         "para assistir a órtese. Sem isto, a varredura corre acelerada")
    ap.add_argument("--simular-ortese", action="store_true", help="com --ortese, não abre a serial")
    a = ap.parse_args()

    if a.listar:
        idx = dados.sujeitos_physionet()
        print(f"PhysioNet ({len(idx)} sujeitos em dados/physionet/):")
        for s, runs in idx.items():
            print(f"  {s}  runs {sorted(runs)}")
        print(f"\nGravações da Cyton (em dados/gravacoes/):")
        for g in dados.sessoes_gravadas():
            print(f"  {g}")
        return

    if not (a.sujeito or a.gravacao):
        sys.exit("Escolha --sujeito ou --gravacao. Use --listar para ver as opções.")

    modelo, amostras, esquema = carregar(a.modelo)
    if amostras != cfg.AMOSTRAS:
        sys.exit(f"O modelo espera {amostras} amostras e o config monta {cfg.AMOSTRAS}. "
                 f"Alinhe config.JANELA_S com o modelo, ou use outro checkpoint.")

    # ---------------- dados ----------------
    contínuo = None
    if a.gravacao:
        pasta = os.path.join(cfg.DIR_GRAVACOES, a.gravacao)
        if not os.path.isdir(pasta):
            sys.exit(f"Não achei {pasta}. Use --listar.")
        saida = dados.janelas_gravacao(pasta)
        if saida is None:
            sys.exit("Nenhuma janela utilizável nessa gravação.")
        X, y, info = saida
        contínuo = dados.sinal_continuo(pasta)
        fonte = f"gravação {a.gravacao}"
        descartadas = sum(1 for i in info if not i["usada"])
    else:
        idx = dados.sujeitos_physionet()
        if a.sujeito not in idx:
            sys.exit(f"Sujeito {a.sujeito} não está em dados/physionet/. Use --listar.")
        runs = [a.run] if a.run else sorted(idx[a.sujeito])
        Xs, ys, info = [], [], []
        for r in runs:
            if r not in idx[a.sujeito]:
                continue
            saida = dados.janelas_edf(idx[a.sujeito][r])
            if saida is None:
                continue
            Xs.append(saida[0]); ys.append(saida[1])
            info += [{"trial": k + 1, "run": r, "usada": True, "t_s": float(t)}
                     for k, t in enumerate(saida[2])]
        if not Xs:
            sys.exit("Nenhuma época válida.")
        X, y = np.concatenate(Xs), np.concatenate(ys)
        descartadas = 0
        fonte = f"PhysioNet {a.sujeito} runs {runs}"

    # Índice de cada janela DENTRO da sua classe, contando de 1. É como o usuário pensa:
    # "o quinto movimento de mão direita", não "a janela número 23".
    usadas = [i for i in (info or []) if i.get("usada", True)]
    ordinal = np.zeros(len(y), int)
    contador = {}
    for k, c in enumerate(y):
        contador[c] = contador.get(c, 0) + 1
        ordinal[k] = contador[c]

    # ---------------- cabeçalho ----------------
    secao("INFERÊNCIA OFFLINE")
    print(f"  fonte ....: {fonte}")
    print(f"  janelas ..: {len(y)}" + (f" ({descartadas} descartadas por não caber)" if descartadas else ""))
    print(f"  modelo ...: {os.path.basename(a.modelo)}")
    print(f"  esquema ..: {esquema.get('nome', '?')} | {amostras} amostras "
          f"({amostras/cfg.TAXA_IA:.2f} s @ {cfg.TAXA_IA} Hz) | canais {cfg.CANAIS}")

    # ---------------- listar os trials ----------------
    if a.trials:
        secao("TRIALS DISPONÍVEIS")
        print(f"  {'#':>4s} {'classe':>7s} {'nº na classe':>13s} {'instante':>10s}   "
              f"use com --trial")
        print("  " + "-" * 62)
        for k in range(len(y)):
            nome = cfg.CLASSES[y[k]]
            t = usadas[k]["t_s"] if k < len(usadas) else float("nan")
            run = f" run {usadas[k]['run']}" if k < len(usadas) and "run" in usadas[k] else ""
            print(f"  {k+1:4d} {nome:>7s} {ordinal[k]:13d} {t:9.1f}s{run}   "
                  f"--trial {nome} {ordinal[k]}")
        print(f"\n  Total por classe: " + " | ".join(
            f"{cfg.CLASSES[c]}: {int((y == c).sum())}" for c in sorted(cfg.CLASSES)))
        return

    # ---------------- um trial só ----------------
    if a.trial:
        classe_pedida, n_str = a.trial
        if classe_pedida not in cfg.CLASSES.values():
            sys.exit(f"Classe '{classe_pedida}' não existe. Use: "
                     f"{', '.join(cfg.CLASSES.values())}")
        try:
            n_pedido = int(n_str)
        except ValueError:
            sys.exit(f"O índice tem que ser um número inteiro, recebi '{n_str}'.")
        cod = {v: k for k, v in cfg.CLASSES.items()}[classe_pedida]
        candidatos = np.where((y == cod) & (ordinal == n_pedido))[0]
        if len(candidatos) == 0:
            disponiveis = int((y == cod).sum())
            sys.exit(f"Não existe o {n_pedido}º '{classe_pedida}' nesta seleção "
                     f"(há {disponiveis}). Use --trials para ver a lista.")
        k = int(candidatos[0])

        pred1, prob1 = prever(modelo, X[k:k + 1])
        p, decisao = prob1[0], cfg.CLASSES[int(pred1[0])]
        t = usadas[k]["t_s"] if k < len(usadas) else float("nan")
        run = usadas[k].get("run") if k < len(usadas) else None

        secao(f"TRIAL: {n_pedido}º '{classe_pedida}'")
        print(f"  janela ....: nº {k+1} da seleção" + (f", run {run}" if run else ""))
        print(f"  instante ..: {t:.1f} s da gravação (centro da janela = a transição)")
        print(f"  trecho ....: {t-cfg.MEIA_JANELA:.1f} s a {t+cfg.MEIA_JANELA:.1f} s")
        print(f"\n  o que era ....: {classe_pedida}")
        print(f"  o que o modelo respondeu: {decisao}")
        print()
        for i in sorted(cfg.CLASSES):
            barra = "#" * int(round(p[i] * 40))
            marca = " <- resposta" if cfg.CLASSES[i] == decisao else ""
            print(f"  {cfg.CLASSES[i]:>7s} {p[i]:6.3f} |{barra:<40s}|{marca}")
        print()
        if a.ortese:
            ort = servo.Ortese(simular=a.simular_ortese)
            try:
                idx_ac = {v: k for k, v in cfg.CLASSES.items()}[cfg.CLASSE_ACIONA]
                if decisao == cfg.CLASSE_ACIONA and p[idx_ac] > cfg.LIMIAR:
                    print(f"  -> comando: FECHAR (p_{cfg.CLASSE_ACIONA}={p[idx_ac]:.3f} "
                          f"passou de {cfg.LIMIAR:g})")
                    ort.fechar()
                    time.sleep(cfg.SEGURA_S)
                else:
                    print(f"  -> comando: nenhum. A órtese fica aberta "
                          f"(p_{cfg.CLASSE_ACIONA}={p[idx_ac]:.3f}, limiar {cfg.LIMIAR:g})")
            finally:
                ort.fechar_conexao()
            print()

        if decisao == classe_pedida:
            print(f"  ACERTOU. Confiança {p[cod]:.1%}"
                  + (f" — acima do limiar de {cfg.LIMIAR:g}, acionaria a órtese."
                     if classe_pedida == cfg.CLASSE_ACIONA and p[cod] > cfg.LIMIAR
                     else f" — abaixo do limiar de {cfg.LIMIAR:g}, NÃO acionaria a órtese."
                     if classe_pedida == cfg.CLASSE_ACIONA else "."))
        else:
            print(f"  ERROU. Era '{classe_pedida}' e respondeu '{decisao}' "
                  f"(confiança na resposta certa: {p[cod]:.1%}).")
        return

    pred, prob = prever(modelo, X)
    m = confusao(y, pred)

    secao("RESULTADO POR JANELA")
    acc, macro, base, maj, p_maj = tabela_metricas(m)
    print()
    print(f"  acurácia .......: {int((pred == y).sum())}/{len(y)} = {acc*100:.1f}%")
    print(f"  macro-F1 .......: {macro:.3f}")
    print(f"  linha de base ..: {base:.3f}  (responder sempre '{maj}' daria {p_maj*100:.1f}%)")
    print(f"  veredito .......: {'acima' if macro > base else 'no nível ou abaixo'} da linha de base")

    secao("MATRIZ DE CONFUSÃO")
    imprimir_confusao(m)

    if a.detalhe:
        secao("JANELA A JANELA")
        nomes = [cfg.CLASSES[i] for i in sorted(cfg.CLASSES)]
        print(f"  {'#':>4s} {'real':>6s} " + " ".join(f"{'p_'+n:>7s}" for n in nomes) + "  decisão")
        for k in range(len(y)):
            marca = "ok" if pred[k] == y[k] else "ERRO"
            print(f"  {k+1:4d} {cfg.CLASSES[y[k]]:>6s} " +
                  " ".join(f"{v:7.3f}" for v in prob[k]) + f"  {cfg.CLASSES[pred[k]]:<6s} {marca}")

    # ---------------- varredura contínua ----------------
    if contínuo is not None:
        secao("JANELA DESLIZANTE (o que a órtese enfrentaria)")
        ort = None
        if a.ortese:
            ort = servo.Ortese(simular=a.simular_ortese)
            if a.ritmo:
                dur = len(contínuo[0]) / float(cfg.TAXA_PLACA)
                print(f"  ritmo real ligado: isto vai levar {dur/60:.1f} min, "
                      f"a duração da gravação. Ctrl-C interrompe.")
        try:
            varredura(modelo, contínuo[0], contínuo[1], a.passo, ort, ritmo=a.ritmo)
        finally:
            if ort is not None:
                ort.fechar_conexao()
    elif a.ortese:
        print("\n  [aviso] --ortese só vale para --gravacao: a varredura contínua precisa")
        print("  de sinal sem cortes, e os EDFs entram aqui já recortados em janelas.")

    secao("COMO LER")
    print(f"  n = {len(y)} janelas. Com poucas dezenas de janelas, nenhuma diferença de alguns")
    print("  pontos é significativa. O macro-F1 é o número a olhar: a acurácia sozinha premia")
    print("  um modelo que responde sempre 'rest', porque descanso é metade dos dados.")


if __name__ == "__main__":
    main()
