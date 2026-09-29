"""
optimize_v3.py
=======================================================================
Barrido de los módulos v3 (M = RSI momentum, O = RSI techo, K = pullback
estocástico) sobre el resto de la estrategia congelado en los defaults v2.

Rejilla conjunta (deduplicada):
    rsiLen   ∈ {7, 10, 14, 21, 28}
    M rsiMin ∈ {off, 40, 45, 50, 55, 60, 65}
    O rsiMax ∈ {off, 65, 70, 75, 80, 85, 90}
    K        ∈ {off} ∪ stLen {5, 9, 14, 21, 28} × stSmooth {1, 3, 5} × stOS {10, 15, 20, 25, 30}

Cada combinación se corre una vez sobre todo el histórico y se mide en:
    IS  = 2018 → 2022 (selección)
    OOS = 2023 → hoy   (validación; NO usar para elegir)
    y en 4 bloques: 2018-20, 2021-22, 2023-24, 2025-26.

Uso:
    python3 btc/optimize_v3.py [--out btc/results/grid_v3.csv.gz] [--procs 4]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from macd_sma800_v3 import Market, P, run, stats  # noqa: E402

IS_END = "2023-01-01"
FOLDS = [("2018", None, "2021-01-01"), ("2021", "2021-01-01", "2023-01-01"),
         ("2023", "2023-01-01", "2025-01-01"), ("2025", "2025-01-01", None)]

_M: Market | None = None


def _init():
    global _M
    _M = Market()


def grid() -> list[P]:
    base = P()
    ks = [None] + list(itertools.product([5, 9, 14, 21, 28], [1, 3, 5], [10, 15, 20, 25, 30]))
    seen, out = set(), []
    for rl, mn, mx, k in itertools.product([7, 10, 14, 21, 28],
                                           [None, 40, 45, 50, 55, 60, 65],
                                           [None, 65, 70, 75, 80, 85, 90], ks):
        if mn is None and mx is None:
            rl = 14  # rsiLen no influye
        key = (rl, mn, mx, k)
        if key in seen:
            continue
        seen.add(key)
        kw = dict(rsiLen=rl, useRsiMom=mn is not None, rsiMin=mn or 50,
                  useRsiCap=mx is not None, rsiMax=mx or 75, useStochPb=k is not None)
        if k:
            kw.update(stLen=k[0], stSmooth=k[1], stOS=k[2])
        out.append(base.__class__(**{**base.__dict__, **kw}))
    return out


def evaluate(p: P) -> dict:
    res = run(_M, p)
    row = dict(rsiLen=p.rsiLen if (p.useRsiMom or p.useRsiCap) else None,
               M=p.rsiMin if p.useRsiMom else None,
               O=p.rsiMax if p.useRsiCap else None,
               stLen=p.stLen if p.useStochPb else None,
               stSmooth=p.stSmooth if p.useStochPb else None,
               stOS=p.stOS if p.useStochPb else None)
    for tag, s, e in [("all", None, None), ("is", None, IS_END), ("oos", IS_END, None)] + FOLDS:
        for k, v in stats(_M, res, s, e).items():
            row[f"{tag}_{k}"] = v
    tr = res["trades"]
    row["n_pb"] = int((tr.type == "pullback").sum()) if len(tr) else 0
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "grid_v3.csv.gz"))
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    g = grid()
    print(f"{len(g)} combinaciones", flush=True)
    with Pool(a.procs, initializer=_init) as pool:
        rows = []
        for i, r in enumerate(pool.imap(evaluate, g, chunksize=16)):
            rows.append(r)
            if i % 1000 == 0:
                print(i, flush=True)
    df = pd.DataFrame(rows)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False, float_format="%.4f")
    print("→", a.out)


if __name__ == "__main__":
    main()
