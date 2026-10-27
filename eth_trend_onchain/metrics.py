"""Métricas, Deflated Sharpe Ratio y walk-forward."""
from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd
from scipy import stats

from config import CostParams, EvalParams, StrategyParams
from strategy import run_backtest


def bars_per_year(idx: pd.DatetimeIndex) -> float:
    hrs = pd.Series(idx).diff().median() / pd.Timedelta(hours=1)
    return 365.0 * 24.0 / hrs                       # cripto cotiza 24/7


def perf(equity: pd.Series, trades: pd.DataFrame | None = None,
         exposure: pd.Series | None = None) -> dict:
    eq = equity.dropna()
    r = eq.pct_change().dropna()
    bpy = bars_per_year(eq.index)
    years = (eq.index[-1] - eq.index[0]) / pd.Timedelta(days=365.25)
    tot = eq.iloc[-1] / eq.iloc[0] - 1
    dd = eq / eq.cummax() - 1
    sd = r.std()
    down = r[r < 0].std()
    out = {
        "retorno_total": tot,
        "CAGR": (1 + tot) ** (1 / years) - 1 if years > 0 and tot > -1 else np.nan,
        "max_DD": dd.min(),
        "sharpe": r.mean() / sd * np.sqrt(bpy) if sd > 0 else np.nan,
        "sortino": r.mean() / down * np.sqrt(bpy) if down > 0 else np.nan,
    }
    out["calmar"] = out["CAGR"] / abs(out["max_DD"]) if out["max_DD"] < 0 else np.nan
    if exposure is not None:
        e = exposure.reindex(eq.index).fillna(0)
        out["tiempo_en_mercado"] = (e != 0).mean()
        out["exposicion_media_abs"] = e.abs()[e != 0].mean() if (e != 0).any() else 0.0
    if trades is not None and not trades.empty:
        t = trades
        win = t["pnl"] > 0
        gp, gl = t.loc[win, "pnl"].sum(), -t.loc[~win, "pnl"].sum()
        top3 = t["pnl"].nlargest(3).sum()
        out.update({
            "n_trades": len(t),
            "win_rate": win.mean(),
            "profit_factor": gp / gl if gl > 0 else np.inf,
            "R_medio": t["R"].mean(),
            "top3_%_beneficio": top3 / t["pnl"].sum() if t["pnl"].sum() > 0 else np.nan,
            "unidades_medias": t["units_max"].mean(),
            "costes_fees": t["fees"].sum(),
            "costes_funding": t["funding"].sum(),
        })
    return out


def buy_hold(close: pd.Series, initial: float) -> pd.Series:
    return initial * close / close.iloc[0]


def per_period_sharpe(equity: pd.Series) -> float:
    r = equity.pct_change().dropna()
    return r.mean() / r.std() if r.std() > 0 else 0.0


def deflated_sharpe(returns: pd.Series, trial_srs: list[float], n_trials: int) -> dict:
    """Bailey & López de Prado (2014). SR por periodo (no anualizado)."""
    r = returns.dropna()
    T = len(r)
    sr = r.mean() / r.std()
    g3 = stats.skew(r)
    g4 = stats.kurtosis(r, fisher=False)
    if n_trials > 1 and len(trial_srs) > 1:
        v = np.var(trial_srs, ddof=1)
        em = 0.5772156649
        sr0 = np.sqrt(v) * ((1 - em) * stats.norm.ppf(1 - 1 / n_trials)
                            + em * stats.norm.ppf(1 - 1 / (n_trials * np.e)))
    else:
        sr0 = 0.0
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr ** 2)
    return {"SR_periodo": sr, "SR0_umbral": sr0, "DSR": float(stats.norm.cdf(z)),
            "n_trials": n_trials, "T": T}


def _slice_metrics(ret: pd.Series) -> dict:
    eq = (1 + ret.fillna(0)).cumprod()
    years = (ret.index[-1] - ret.index[0]) / pd.Timedelta(days=365.25)
    tot = eq.iloc[-1] - 1
    return {"ret": tot,
            "cagr": (1 + tot) ** (1 / years) - 1 if years > 0 and tot > -1 else -1,
            "dd": (eq / eq.cummax() - 1).min()}


def walkforward(df: pd.DataFrame, base: StrategyParams, grid: dict, cost: CostParams,
                ev: EvalParams, variants=(False, True)) -> dict:
    """WF rodante: elige en train (máx. CAGR con DD <= max_dd; si ninguno cumple,
    máx. Calmar) y evalúa en test. Las señales son causales, así que se corre la
    historia completa una vez por configuración y se recortan los retornos."""
    keys = list(grid)
    cfgs = [dict(zip(keys, v)) for v in product(*grid.values())]
    runs = {}
    for v in variants:
        for k, cfg in enumerate(cfgs):
            p = replace(base, **cfg, use_onchain=v)
            res = run_backtest(df, p, cost, ev.initial_equity)
            runs[(v, k)] = res.equity.pct_change().fillna(0)

    trial_srs = [r.mean() / r.std() for r in runs.values() if r.std() > 0]
    n_trials = len(runs)

    warm = max(base.atr_len, max(grid.get("sma_regime", [base.sma_regime])),
               max(grid.get("donchian_entry", [base.donchian_entry])))
    start = df.index[min(warm, len(df) - 1)]
    end = df.index[-1]
    folds = []
    t0 = start + pd.DateOffset(years=ev.train_years)
    while t0 + pd.DateOffset(years=ev.test_years) <= end + pd.Timedelta(days=1):
        folds.append((t0 - pd.DateOffset(years=ev.train_years), t0,
                      t0 + pd.DateOffset(years=ev.test_years)))
        t0 = t0 + pd.DateOffset(years=ev.test_years)

    bh = df["close"].pct_change().fillna(0)
    out = {"folds": {}, "oos": {}, "dsr": {}, "cfgs": cfgs}
    for v in variants:
        rows, oos = [], []
        for a, b, cc in folds:
            scores = []
            for k in range(len(cfgs)):
                m = _slice_metrics(runs[(v, k)][(runs[(v, k)].index >= a) & (runs[(v, k)].index < b)])
                ok = m["dd"] >= -ev.max_dd
                calmar = m["cagr"] / abs(m["dd"]) if m["dd"] < 0 else m["cagr"]
                scores.append((ok, m["cagr"] if ok else -np.inf, calmar, k))
            if any(s[0] for s in scores):
                best = max(scores, key=lambda s: s[1])[3]
            else:
                best = max(scores, key=lambda s: s[2])[3]
            rb = runs[(v, best)]
            r_test = rb[(rb.index >= b) & (rb.index < cc)]
            mt = _slice_metrics(r_test)
            mb = _slice_metrics(bh[(bh.index >= b) & (bh.index < cc)])
            rows.append({"train": f"{a:%Y-%m}→{b:%Y-%m}", "test": f"{b:%Y-%m}→{cc:%Y-%m}",
                         **{f"p_{kk}": cfgs[best][kk] for kk in keys},
                         "OOS_ret": mt["ret"], "OOS_DD": mt["dd"], "BH_ret": mb["ret"],
                         "BH_DD": mb["dd"], "supera_BH": mt["ret"] > mb["ret"],
                         "DD_ok": mt["dd"] >= -ev.max_dd})
            oos.append(r_test)
        label = "B_con_onchain" if v else "A_sin_onchain"
        out["folds"][label] = pd.DataFrame(rows)
        if oos:
            stitched = pd.concat(oos)
            out["oos"][label] = stitched
            out["dsr"][label] = deflated_sharpe(stitched, trial_srs, n_trials)
    if folds:
        bh_oos = bh[(bh.index >= folds[0][1]) & (bh.index < folds[-1][2])]
        out["oos"]["buy_hold"] = bh_oos
    return out
