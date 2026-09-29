"""
analyze_v3.py
=======================================================================
Resume btc/results/grid_v3.csv (salida de optimize_v3.py):
  1. Baseline (v3 con todo apagado == v2).
  2. Cada módulo aislado (M, O, K) por parámetro, IS vs OOS.
  3. Selección por IS y su resultado OOS; correlación de rangos IS→OOS.
  4. Sharpe deflactado (Bailey & López de Prado) del mejor IS contando
     todas las pruebas de la rejilla.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parent / "results" / "grid_v3.csv"
pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

COLS = ["all_net", "all_maxdd", "all_sharpe", "all_trades",
        "is_net", "is_maxdd", "is_sharpe", "is_trades",
        "oos_net", "oos_maxdd", "oos_sharpe", "oos_trades"]


def _norm_cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _norm_ppf(p):
    # Acklam, suficiente para esto
    from statistics import NormalDist
    return NormalDist().inv_cdf(p)


def deflated_sharpe(sr_ann, n_days, n_trials, sr_var_ann, skew=0.0, kurt=3.0):
    """DSR con SR anualizados → por observación diaria."""
    sr = sr_ann / math.sqrt(365)
    v = sr_var_ann / 365
    g = 0.5772156649
    e_max = math.sqrt(v) * ((1 - g) * _norm_ppf(1 - 1 / n_trials) + g * _norm_ppf(1 - 1 / (n_trials * math.e)))
    z = (sr - e_max) * math.sqrt(n_days - 1) / math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return _norm_cdf(z), e_max * math.sqrt(365)


def main():
    df = pd.read_csv(R)
    off = df.M.isna() & df.O.isna() & df.stLen.isna()
    base = df[off].iloc[0]
    print("=== Baseline (M/O/K apagados) ===")
    print(base[COLS].to_string())

    def delta(sub, by):
        g = sub.groupby(by)[COLS].mean()
        for k in ["all_net", "is_net", "oos_net", "all_sharpe", "is_sharpe", "oos_sharpe"]:
            g["d_" + k] = g[k] - base[k]
        return g.round(2)

    onlyM = df[df.M.notna() & df.O.isna() & df.stLen.isna()]
    onlyO = df[df.M.isna() & df.O.notna() & df.stLen.isna()]
    onlyK = df[df.M.isna() & df.O.isna() & df.stLen.notna()]
    print("\n=== Solo M (RSI mínimo) ===")
    print(onlyM.pivot_table(index="rsiLen", columns="M", values="all_sharpe").round(2))
    print("OOS Sharpe"); print(onlyM.pivot_table(index="rsiLen", columns="M", values="oos_sharpe").round(2))
    print("Operaciones"); print(onlyM.pivot_table(index="rsiLen", columns="M", values="all_trades"))
    print("\n=== Solo O (RSI techo) ===")
    print(onlyO.pivot_table(index="rsiLen", columns="O", values="all_sharpe").round(2))
    print("OOS Sharpe"); print(onlyO.pivot_table(index="rsiLen", columns="O", values="oos_sharpe").round(2))
    print("Operaciones"); print(onlyO.pivot_table(index="rsiLen", columns="O", values="all_trades"))
    print("\n=== Solo K (pullback estocástico) — Sharpe total, media sobre suavizado ===")
    print(onlyK.pivot_table(index="stLen", columns="stOS", values="all_sharpe").round(2))
    print("IS Sharpe"); print(onlyK.pivot_table(index="stLen", columns="stOS", values="is_sharpe").round(2))
    print("OOS Sharpe"); print(onlyK.pivot_table(index="stLen", columns="stOS", values="oos_sharpe").round(2))
    print("Suavizado (media)"); print(onlyK.groupby("stSmooth")[["all_sharpe", "is_sharpe", "oos_sharpe", "all_maxdd", "n_pb"]].mean().round(2))
    print("Entradas pullback"); print(onlyK.pivot_table(index="stLen", columns="stOS", values="n_pb").round(1))

    print("\n=== % de combinaciones que mejoran al baseline ===")
    for name, sub in [("M", onlyM), ("O", onlyO), ("K", onlyK), ("todas", df[~off])]:
        print(f"{name:6s} n={len(sub):5d}  IS Sharpe>{base.is_sharpe:.2f}: {(sub.is_sharpe > base.is_sharpe).mean()*100:5.1f}%"
              f"  OOS Sharpe>{base.oos_sharpe:.2f}: {(sub.oos_sharpe > base.oos_sharpe).mean()*100:5.1f}%"
              f"  ambos: {((sub.is_sharpe > base.is_sharpe) & (sub.oos_sharpe > base.oos_sharpe)).mean()*100:5.1f}%")

    rho = df[["is_sharpe", "oos_sharpe"]].rank().corr().iloc[0, 1]
    print(f"\nCorrelación de rangos Sharpe IS→OOS (toda la rejilla): {rho:.3f}")

    print("\n=== Top 15 por Sharpe IS (2018-22) y su OOS (2023-26) ===")
    keys = ["rsiLen", "M", "O", "stLen", "stSmooth", "stOS"]
    top = df.sort_values("is_sharpe", ascending=False).head(15)
    print(top[keys + COLS].round(2).to_string(index=False))
    print("\n=== Top 15 por Sharpe total 2018-26 ===")
    print(df.sort_values("all_sharpe", ascending=False).head(15)[keys + COLS].round(2).to_string(index=False))

    folds = [c for c in df.columns if c.endswith("_sharpe") and c[:4].isdigit()]
    df["min_fold"] = df[folds].min(axis=1)
    print("\n=== Top 15 por peor bloque (robustez: max del min Sharpe en 2018-20/21-22/23-24/25-26) ===")
    print(df.sort_values("min_fold", ascending=False).head(15)[keys + folds + ["all_sharpe", "all_maxdd", "all_trades"]].round(2).to_string(index=False))
    print("Baseline:", base[folds].round(2).to_dict())

    best = df.sort_values("all_sharpe", ascending=False).iloc[0]
    n_days = 365 * 8.5
    dsr, emax = deflated_sharpe(best.all_sharpe, n_days, len(df), df.all_sharpe.var())
    print(f"\nDSR del mejor total (SR={best.all_sharpe:.2f}, N={len(df)}): {dsr:.3f}; SR máximo esperado por azar={emax:.2f}")
    dsr_b, _ = deflated_sharpe(base.all_sharpe, n_days, len(df), df.all_sharpe.var())
    print(f"Mejora del mejor sobre baseline: {best.all_sharpe - base.all_sharpe:+.2f} vs dispersión de la rejilla σ={df.all_sharpe.std():.2f}")


if __name__ == "__main__":
    main()
