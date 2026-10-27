"""
CLI para evaluar la estrategia ETH trend following + on-chain.

  python main.py prices      --exchange okx --timeframe 1d --since 2019-01-01 --out data/eth_1d.csv
  python main.py onchain     --project MI_PROYECTO_GCP --start 2016-01-01 \
                             --exchanges exchange_addresses.csv --out data/onchain_daily.csv
  python main.py check       --prices data/eth_1d.csv --onchain data/onchain_daily.csv
  python main.py backtest    --prices data/eth_1d.csv --onchain data/onchain_daily.csv --html out/bt.html
  python main.py walkforward --prices data/eth_1d.csv --onchain data/onchain_daily.csv
  python main.py demo        # datos SINTÉTICOS, solo para comprobar que todo funciona
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from config import GRID, CostParams, EvalParams, OnchainParams, StrategyParams
from data import (bar_delta, fetch_ohlcv_ccxt, funding_per_bar, load_funding_csv,
                  load_ohlcv_csv)
from metrics import buy_hold, perf, walkforward
from onchain import (align_to_bars, build_signal, fetch_onchain_bigquery,
                     load_onchain_csv, onchain_check)
from strategy import run_backtest

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)


def prepare(prices: str, onchain: str | None, funding: str | None,
            op: OnchainParams, start: str | None = None, end: str | None = None):
    df = load_ohlcv_csv(prices)
    td = bar_delta(df.index)
    sig = None
    if onchain:
        sig = build_signal(load_onchain_csv(onchain), op)
        df = df.join(align_to_bars(sig, df.index, td, op.lag_hours))
    if funding:
        df["funding"] = funding_per_bar(load_funding_csv(funding), df.index)
    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index < pd.Timestamp(end, tz="UTC")]
    return df, sig


def fmt(d: dict) -> pd.Series:
    s = pd.Series(d, dtype=object)
    for k, v in s.items():
        if isinstance(v, (float, np.floating)):
            pct = any(t in k for t in ("retorno", "CAGR", "DD", "win", "tiempo", "top3", "ret"))
            s[k] = f"{v:.1%}" if pct else f"{v:,.2f}"
    return s


def cmd_prices(a):
    df = fetch_ohlcv_ccxt(a.exchange, a.symbol, a.timeframe, a.since)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index_label="time")
    print(f"{len(df)} velas {a.timeframe} {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d} en {a.out}")


def cmd_onchain(a):
    df = fetch_onchain_bigquery(a.project, a.start, a.exchanges, a.max_gb, a.yes)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} días guardados en {a.out}")


def cmd_check(a):
    op = OnchainParams()
    px = load_ohlcv_csv(a.prices)
    daily = px["close"].resample("1D").last().dropna() if bar_delta(px.index) < pd.Timedelta(days=1) \
        else px["close"]
    sig = build_signal(load_onchain_csv(a.onchain), op)
    res = onchain_check(daily, sig, op.lag_hours)
    print("\n¿La alerta on-chain anticipa caídas? (dd = caída máxima en el horizonte)\n")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\nDías con alerta: {sig['alert'].mean():.1%} del total. "
          "p_valor_dd < 0,05 sugiere que tras la alerta las caídas son mayores que por azar.")


def run_ab(df, base, cost, ev):
    results = {}
    variants = [False, True] if "oc_alert" in df else [False]
    for v in variants:
        name = "B_con_onchain" if v else "A_sin_onchain"
        results[name] = run_backtest(df, replace(base, use_onchain=v), cost, ev.initial_equity)
    return results


def cmd_backtest(a):
    op, cost, ev = OnchainParams(), CostParams(), EvalParams()
    base = StrategyParams()
    df, _ = prepare(a.prices, a.onchain, a.funding, op, a.start, a.end)
    results = run_ab(df, base, cost, ev)
    bh = buy_hold(df["close"], ev.initial_equity)
    table = {name: fmt(perf(r.equity, r.trades, r.exposure)) for name, r in results.items()}
    table["buy_hold"] = fmt(perf(bh))
    print(f"\nETH {df.index[0]:%Y-%m-%d} → {df.index[-1]:%Y-%m-%d} | barras: {bar_delta(df.index)}")
    order = list(next(iter(table.values())).index)
    print(pd.DataFrame(table).reindex(order).fillna("").to_string())
    for name, r in results.items():
        dd = r.equity / r.equity.cummax() - 1
        flag = "OK" if dd.min() >= -ev.max_dd else "SUPERA"
        print(f"{name}: DD máx {dd.min():.1%} → criterio {ev.max_dd:.0%}: {flag}")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    for name, r in results.items():
        r.trades.to_csv(out / f"trades_{name}.csv", index=False)
    (out / "params.json").write_text(json.dumps(
        {"strategy": asdict(base), "onchain": asdict(op), "cost": asdict(cost)},
        indent=2, default=str))
    if a.html:
        from report import plot_html
        plot_html(df, results, bh, a.html, op.alert_z)
        print(f"Gráfico: {a.html}")


def cmd_walkforward(a):
    op, cost, ev = OnchainParams(), CostParams(), EvalParams()
    df, _ = prepare(a.prices, a.onchain, a.funding, op, a.start, a.end)
    variants = (False, True) if "oc_alert" in df else (False,)
    wf = walkforward(df, StrategyParams(), GRID, cost, ev, variants)
    for label, t in wf["folds"].items():
        print(f"\n=== {label} ===")
        print(t.to_string(index=False, float_format=lambda x: f"{x:.1%}"))
        print(f"Folds que superan B&H: {t['supera_BH'].sum()}/{len(t)} | "
              f"folds con DD ≤ {ev.max_dd:.0%}: {t['DD_ok'].sum()}/{len(t)}")
    print("\n=== OOS encadenado ===")
    tab = {}
    for label, r in wf["oos"].items():
        eq = ev.initial_equity * (1 + r).cumprod()
        tab[label] = fmt(perf(eq))
    print(pd.DataFrame(tab).reindex(list(next(iter(tab.values())).index)).to_string())
    print("\n=== Deflated Sharpe (trials = rejilla × variantes) ===")
    for label, d in wf["dsr"].items():
        print(f"{label}: DSR={d['DSR']:.3f}  SR/periodo={d['SR_periodo']:.4f}  "
              f"umbral={d['SR0_umbral']:.4f}  trials={d['n_trials']}  T={d['T']}")
    print("DSR > 0,95 ≈ el Sharpe OOS sobrevive al número de pruebas realizadas.")


def make_synthetic(out: Path, seed: int = 11):
    """Precios con regímenes alcista/bajista y on-chain que se dispara a veces antes
    de los giros bajistas. Solo para probar el código: NO dice nada de ETH real."""
    rng = np.random.default_rng(seed)
    days = pd.date_range("2017-01-01", "2026-09-30", freq="1D", tz="UTC")
    n = len(days)
    regime = np.zeros(n, int)
    for i in range(1, n):
        regime[i] = regime[i - 1] if rng.random() > 1 / 120 else 1 - regime[i - 1]
    mu = np.where(regime == 0, 0.004, -0.003)
    r = mu + rng.standard_t(4, n) * 0.035
    close = 100 * np.exp(np.cumsum(r))
    open_ = np.r_[close[0], close[:-1]] * np.exp(rng.normal(0, 0.003, n))
    span = np.abs(rng.normal(0, 0.025, n)) * close
    high = np.maximum(open_, close) + span * rng.random(n)
    low = np.minimum(open_, close) - span * rng.random(n)
    px = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": rng.lognormal(10, 0.5, n)}, index=days)
    px.to_csv(out / "demo_prices_1d.csv", index_label="time")
    turn = np.r_[np.diff(regime), 0] == 1                      # giro a bajista mañana
    pre = pd.Series(turn[::-1]).rolling(10, min_periods=1).max()[::-1].values.astype(bool)
    boost = np.where(pre & (rng.random(n) < 0.6), 3.0, 1.0)
    oc = pd.DataFrame({
        "date": days.strftime("%Y-%m-%d"),
        "whale_volume": rng.lognormal(12, 0.4, n) * boost,
        "vdd": rng.lognormal(14, 0.5, n) * boost,
        "exch_inflow": rng.lognormal(11, 0.3, n) * boost,
        "exch_outflow": rng.lognormal(11, 0.3, n),
    })
    oc.to_csv(out / "demo_onchain_daily.csv", index=False)


def cmd_demo(a):
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    make_synthetic(out)
    print("*** DATOS SINTÉTICOS: solo comprueban que el código corre ***")
    ns = argparse.Namespace(prices=str(out / "demo_prices_1d.csv"),
                            onchain=str(out / "demo_onchain_daily.csv"), funding=None,
                            start=None, end=None, outdir=str(out),
                            html=str(out / "demo_backtest.html"))
    cmd_check(ns)
    cmd_backtest(ns)
    cmd_walkforward(ns)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("prices")
    s.add_argument("--exchange", default="okx", choices=["okx", "binance"])
    s.add_argument("--symbol", default=None)
    s.add_argument("--timeframe", default="1d", choices=["4h", "6h", "1d"])
    s.add_argument("--since", default="2017-01-01")
    s.add_argument("--out", default="data/eth_1d.csv")
    s.set_defaults(fn=cmd_prices)

    s = sub.add_parser("onchain")
    s.add_argument("--project", required=True, help="ID de tu proyecto de Google Cloud")
    s.add_argument("--start", default="2016-01-01")
    s.add_argument("--exchanges", default=None, help="CSV con columna 'address'")
    s.add_argument("--max-gb", type=float, default=600.0)
    s.add_argument("--yes", action="store_true")
    s.add_argument("--out", default="data/onchain_daily.csv")
    s.set_defaults(fn=cmd_onchain)

    s = sub.add_parser("check")
    s.add_argument("--prices", required=True)
    s.add_argument("--onchain", required=True)
    s.set_defaults(fn=cmd_check)

    for name, fn in (("backtest", cmd_backtest), ("walkforward", cmd_walkforward)):
        s = sub.add_parser(name)
        s.add_argument("--prices", required=True)
        s.add_argument("--onchain", default=None)
        s.add_argument("--funding", default=None, help="CSV time,rate")
        s.add_argument("--start", default=None)
        s.add_argument("--end", default=None)
        s.add_argument("--outdir", default="out")
        if name == "backtest":
            s.add_argument("--html", default="out/backtest.html")
        s.set_defaults(fn=fn)

    s = sub.add_parser("demo")
    s.add_argument("--outdir", default="out_demo")
    s.set_defaults(fn=cmd_demo)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
