"""
macd_sma800_6h_backtest.py
=======================================================================
Réplica en Python de pine/macd_d1_sma800_6h.pine (MACD D1 régimen +
SMA800 6H entrada + Chandelier ATR salida) para optimizarla fuera de
TradingView. Ver notes/2026-09-26-macd-sma800-6h-btc-optimization.md.

Semántica del broker de TradingView replicada:
  - el script corre al cierre de cada vela; entradas/cierres a mercado se
    llenan en la apertura de la vela siguiente (process_orders_on_close=false)
  - stops (strategy.exit) activos durante la vela siguiente; si la apertura
    ya está más allá del stop, se llena a la apertura (hueco)
  - request.security("D", x[1], lookahead_on) = valor de la vela diaria
    CERRADA anterior (días UTC, como en los exchanges cripto)
  - 100 % del equity por operación, comisión 0,0035 % por lado, 1 tick de
    deslizamiento
No replica el cierre parcial (P): la optimización se hizo con P apagado.

Uso:
    python3 macd_sma800_6h_backtest.py fetch               # descarga BTC 6h/1d
    python3 macd_sma800_6h_backtest.py eval [--sl 3 --ch 8 --dir long ...]
    python3 macd_sma800_6h_backtest.py optimize            # rejilla + robustez

Los datos se guardan en data/btc/ (no versionados, ver .gitignore).
Requiere numpy + pandas; numba es opcional (acelera ~50x la rejilla).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from numba import njit
except ImportError:  # numba opcional
    def njit(*a, **k):
        return (lambda f: f) if not (len(a) == 1 and callable(a[0])) else a[0]

DATA_DIR = Path(__file__).resolve().parent / "data" / "btc"
SOURCES = ("binance", "bitstamp", "coinbase")
T0 = pd.Timestamp("2020-01-03", tz="UTC")       # inicio del backtest en TradingView
T_PRE = pd.Timestamp("2018-06-01", tz="UTC")    # holdout anterior a la ventana de TV
T_MID = pd.Timestamp("2023-07-01", tz="UTC")    # corte walk-forward H1 / H2
START = "2017-01-01"


# ----------------------------------------------------------------------
# Descarga
# ----------------------------------------------------------------------
def _get(url):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return json.load(urllib.request.urlopen(req, timeout=30))


def _frame(rows, cols):
    df = pd.DataFrame(rows, columns=cols)
    df["t"] = pd.to_datetime(df["t"].astype("int64"), unit=df.attrs.get("unit", "s"), utc=True)
    df = df.set_index("t").sort_index()
    return df[~df.index.duplicated()][["open", "high", "low", "close", "volume"]].astype(float)


def fetch_binance(seconds):
    iv = {21600: "6h", 86400: "1d"}[seconds]
    rows, s = [], int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    while True:
        d = _get(f"https://data-api.binance.vision/api/v3/klines?symbol=BTCUSDT"
                 f"&interval={iv}&startTime={s}&limit=1000")
        if not d:
            break
        rows += [r[:6] for r in d]
        s = d[-1][0] + 1
        if len(d) < 1000:
            break
    rows = [[r[0] // 1000] + r[1:] for r in rows]
    return _frame(rows, ["t", "open", "high", "low", "close", "volume"])


def fetch_bitstamp(seconds):
    rows, s, end = [], int(pd.Timestamp(START, tz="UTC").timestamp()), int(time.time())
    while s < end:
        d = _get(f"https://www.bitstamp.net/api/v2/ohlc/btcusd/?step={seconds}&limit=1000&start={s}")
        d = d["data"]["ohlc"]
        if not d:
            break
        rows += [[r["timestamp"], r["open"], r["high"], r["low"], r["close"], r["volume"]] for r in d]
        s = int(d[-1]["timestamp"]) + seconds
        if len(d) < 1000:
            break
    return _frame(rows, ["t", "open", "high", "low", "close", "volume"])


def fetch_coinbase(seconds):
    rows, end, start0 = [], pd.Timestamp.now(tz="UTC"), pd.Timestamp(START, tz="UTC")
    while end > start0:
        st = end - pd.Timedelta(seconds=seconds * 300)
        d = _get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?granularity={seconds}"
                 f"&start={st.isoformat()}&end={end.isoformat()}")
        if not d:
            break
        rows += d
        end = st
        time.sleep(0.2)
    return _frame(rows, ["t", "low", "high", "open", "close", "volume"])


def fetch_all():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fns = {"binance": fetch_binance, "bitstamp": fetch_bitstamp, "coinbase": fetch_coinbase}
    for src, fn in fns.items():
        for tag, sec in (("6h", 21600), ("1d", 86400)):
            df = fn(sec)
            df.to_csv(DATA_DIR / f"{src}_{tag}.csv")
            print(f"{src} {tag}: {len(df)} velas {df.index[0]} -> {df.index[-1]}")


def load(src):
    rd = lambda tag: pd.read_csv(DATA_DIR / f"{src}_{tag}.csv", index_col=0, parse_dates=True)
    return rd("6h"), rd("1d")


# ----------------------------------------------------------------------
# Indicadores con la semántica de Pine (ta.rma / ta.ema sembradas con SMA)
# ----------------------------------------------------------------------
def rma(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    i0 = int(np.argmax(~np.isnan(x)))
    if len(x) - i0 < n:
        return out
    out[i0 + n - 1] = x[i0:i0 + n].mean()
    for i in range(i0 + n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def ema(x, n):
    x = np.asarray(x, float)
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    a = 2 / (n + 1)
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def true_range(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def adx(h, l, c, n):
    up, dn = np.r_[np.nan, np.diff(h)], -np.r_[np.nan, np.diff(l)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = mdm[0] = np.nan
    tr = rma(true_range(h, l, c), n)
    p = pd.Series(100 * rma(pdm, n) / tr).ffill().values
    m = pd.Series(100 * rma(mdm, n) / tr).ffill().values
    s = p + m
    return 100 * rma(np.abs(p - m) / np.where(s == 0, 1, s), n)


def features(h6, d1, smaLen=800, adxLen=14, atrLen=14, slopeLen=20, reLen=20, macdFast=12, macdSlow=26):
    o, h, l, c = (h6[k].values for k in ("open", "high", "low", "close"))
    day = h6.index.floor("D")
    prev_day = lambda v: pd.Series(v, index=d1.index).shift(1).reindex(day).values
    dc = d1.close.values
    sma = pd.Series(c).rolling(smaLen).mean().values
    return dict(
        o=o, h=h, l=l, c=c, sma=sma,
        smaPrev=np.r_[np.full(slopeLen, np.nan), sma[:-slopeLen]],
        atr6=rma(true_range(h, l, c), atrLen),
        atrD=prev_day(rma(true_range(d1.high.values, d1.low.values, dc), atrLen)),
        macdD=prev_day(ema(dc, macdFast) - ema(dc, macdSlow)),
        adx=adx(h, l, c, adxLen),
        hiRe=pd.Series(h).rolling(reLen).max().shift(1).values,
        loRe=pd.Series(l).rolling(reLen).min().shift(1).values,
    )


# ----------------------------------------------------------------------
# Broker emulator
# ----------------------------------------------------------------------
@njit(cache=True)
def _run(o, h, l, c, sma, atr, adxv, macdD, smaPrev, hiRe, loRe, tradeStart,
         slMult, chMult, useAdx, adxThr, useSlope, useReentry, exitOnRegime, dirMode,
         comm, slip, cap):
    n = len(c)
    pos = 0.0; avg = 0.0; cash = cap; prevPos = 0.0
    pend = 0; pendQty = 0.0          # pend: 1 largo, -1 corto, 2 cerrar
    stopL = np.nan; stopS = np.nan
    riskL = np.nan; trailL = np.nan; hhv = np.nan
    riskS = np.nan; trailS = np.nan; llv = np.nan
    eq = np.full(n, cap)
    pnl = np.zeros(n); ent = np.zeros(n, np.int64); ext = np.zeros(n, np.int64); side = np.zeros(n)
    ntr = 0; entryBar = -1
    for i in range(n):
        # --- broker: órdenes emitidas al cierre de i-1 ---
        exitPx = np.nan
        if pend == 2 and pos != 0:
            exitPx = o[i] - slip if pos > 0 else o[i] + slip
        elif pend == 1 or pend == -1:
            d = 1.0 if pend == 1 else -1.0
            if pos != 0:
                exitPx = o[i] + slip * d
        if not np.isnan(exitPx):
            p = pos * (exitPx - avg) - abs(pos) * exitPx * comm
            cash += p
            pnl[ntr] = p - abs(pos) * avg * comm; ent[ntr] = entryBar; ext[ntr] = i
            side[ntr] = 1.0 if pos > 0 else -1.0; ntr += 1
            pos = 0.0
        if pend == 1 or pend == -1:
            d = 1.0 if pend == 1 else -1.0
            avg = o[i] + slip * d
            pos = d * pendQty
            cash -= abs(pos) * avg * comm
            entryBar = i
        pend = 0
        hit = False
        if pos > 0 and not np.isnan(stopL) and l[i] <= stopL:
            exitPx = (o[i] if o[i] <= stopL else stopL) - slip; hit = True
        elif pos < 0 and not np.isnan(stopS) and h[i] >= stopS:
            exitPx = (o[i] if o[i] >= stopS else stopS) + slip; hit = True
        if hit:
            p = pos * (exitPx - avg) - abs(pos) * exitPx * comm
            cash += p
            pnl[ntr] = p - abs(pos) * avg * comm; ent[ntr] = entryBar; ext[ntr] = i
            side[ntr] = 1.0 if pos > 0 else -1.0; ntr += 1
            pos = 0.0
        stopL = np.nan; stopS = np.nan
        equity = cash + pos * (c[i] - avg)
        eq[i] = equity
        # --- script al cierre de la vela i ---
        if i == 0 or np.isnan(sma[i - 1]) or np.isnan(macdD[i]) or np.isnan(atr[i]):
            prevPos = pos
            continue
        regL = macdD[i] > 0; regS = macdD[i] < 0
        adxOK = (not useAdx) or adxv[i] > adxThr
        slL = (not useSlope) or sma[i] > smaPrev[i]
        slS = (not useSlope) or sma[i] < smaPrev[i]
        crossUp = c[i] > sma[i] and c[i - 1] <= sma[i - 1]
        crossDn = c[i] < sma[i] and c[i - 1] >= sma[i - 1]
        trigL = crossUp or (useReentry and c[i] > sma[i] and c[i] > hiRe[i])
        trigS = crossDn or (useReentry and c[i] < sma[i] and c[i] < loRe[i])
        live = i >= tradeStart
        entryL = live and trigL and regL and dirMode != 2 and pos <= 0 and slL and adxOK
        entryS = live and trigS and regS and dirMode != 1 and pos >= 0 and slS and adxOK
        if entryL:
            pend = 1; pendQty = equity / c[i]
            riskL = slMult * atr[i]; trailL = c[i] - riskL; hhv = c[i]
        if entryS:
            pend = -1; pendQty = equity / c[i]
            riskS = slMult * atr[i]; trailS = c[i] + riskS; llv = c[i]
        if pos > 0:
            if prevPos <= 0:
                trailL = avg - riskL; hhv = h[i]
            else:
                hhv = max(hhv, h[i])
            trailL = max(trailL, hhv - chMult * atr[i])
        if pos < 0:
            if prevPos >= 0:
                trailS = avg + riskS; llv = l[i]
            else:
                llv = min(llv, l[i])
            trailS = min(trailS, llv + chMult * atr[i])
        if entryL or pos > 0:
            stopL = trailL
        if entryS or pos < 0:
            stopS = trailS
        if exitOnRegime and ((pos > 0 and not regL and not entryS) or (pos < 0 and not regS and not entryL)):
            pend = 2
        prevPos = pos
    return eq, pnl[:ntr], ent[:ntr], ext[:ntr], side[:ntr]


DEFAULTS = dict(smaLen=800, adxLen=14, slopeLen=20, reLen=20, sl=3.0, ch=8.0, adx=True, adxThr=20.0,
                slope=False, reentry=True, atrD=True, regExit=True, dir="long")
DIRS = {"both": 0, "long": 1, "short": 2}


class Backtester:
    def __init__(self, src):
        self.h6, self.d1 = load(src)
        self._cache = {}

    def feats(self, p):
        k = (p["smaLen"], p["adxLen"], p["slopeLen"], p["reLen"])
        if k not in self._cache:
            self._cache[k] = features(self.h6, self.d1, smaLen=k[0], adxLen=k[1], slopeLen=k[2], reLen=k[3])
        return self._cache[k]

    def run(self, p, t0=T0, t1=None, cap=10000.0):
        p = dict(DEFAULTS, **p)
        F = self.feats(p)
        idx = self.h6.index
        ts = int(np.searchsorted(idx, t0))
        te = len(idx) if t1 is None else int(np.searchsorted(idx, t1))
        a = lambda k: F[k][:te]
        eq, pnl, ent, ext, side = _run(
            a("o"), a("h"), a("l"), a("c"), a("sma"), a("atrD") if p["atrD"] else a("atr6"), a("adx"),
            a("macdD"), a("smaPrev"), a("hiRe"), a("loRe"), ts, p["sl"], p["ch"], p["adx"], p["adxThr"],
            p["slope"], p["reentry"], p["regExit"], DIRS[p["dir"]], 0.0035 / 100, 0.01, cap)
        e = eq[ts:]
        dd = ((np.maximum.accumulate(e) - e) / np.maximum.accumulate(e)).max() * 100
        net = (e[-1] / cap - 1) * 100
        yrs = max((idx[te - 1] - idx[ts]).days / 365.25, 1e-9)
        gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
        cagr = ((1 + net / 100) ** (1 / yrs) - 1) * 100
        trades = pd.DataFrame(dict(entry=idx[ent], exit=idx[ext], side=side, pnl=pnl))
        return dict(net=net, cagr=cagr, dd=dd, calmar=cagr / dd if dd > 0 else np.nan, trades=len(pnl),
                    wins=int((pnl > 0).sum()), pf=gp / gl if gl > 0 else np.inf), trades


# ----------------------------------------------------------------------
# Optimización: rejilla + meseta de robustez + walk-forward + holdout
# ----------------------------------------------------------------------
GRID = dict(sl=[1.5, 2, 2.5, 3, 4], ch=[4, 6, 8], adxThr=[15, 20, 25, 30], adxLen=[14, 20],
            smaLen=[600, 800, 1000], slopeLen=[10, 20, 40], reLen=[10, 20, 40],
            slope=[True, False], reentry=[True, False], atrD=[True, False], regExit=[True, False],
            dir=["both", "long"])


def optimize(out_csv):
    bts = {s: Backtester(s) for s in SOURCES}
    keys, rows = list(GRID), []
    for vals in itertools.product(*GRID.values()):
        p = dict(zip(keys, vals))
        if (not p["slope"] and p["slopeLen"] != 20) or (not p["reentry"] and p["reLen"] != 20):
            continue  # parámetro inactivo: no duplicar combinaciones
        r = dict(p)
        for s, bt in bts.items():
            st, _ = bt.run(p)
            r.update({f"{s}_net": st["net"], f"{s}_dd": st["dd"], f"{s}_calmar": st["calmar"], f"{s}_n": st["trades"]})
        bt = bts["bitstamp"]
        r["pre_net"] = bt.run(p, t0=T_PRE, t1=T0)[0]["net"]
        h1 = bt.run(p, t1=T_MID)[0]
        r["h1_net"], r["h1_calmar"] = h1["net"], h1["calmar"]
        r["h2_net"] = bt.run(p, t0=T_MID)[0]["net"]
        rows.append(r)
    df = pd.DataFrame(rows)
    df["calmar_min"] = df[[f"{s}_calmar" for s in SOURCES]].min(axis=1)
    df.to_csv(out_csv, index=False)
    top = df.sort_values("h1_calmar", ascending=False).head(200)
    print(f"{len(df)} combinaciones -> {out_csv}")
    print("Walk-forward: top-200 por Calmar en H1 (2020-01..2023-06) quedan en el percentil "
          f"{df.h2_net.rank(pct=True)[top.index].median():.0%} de H2 (2023-07..hoy)")
    print("Moda de parámetros del top-200 H1:")
    print(top[keys].agg(lambda s: s.value_counts().idxmax()).to_string())
    print("\nTop 15 por Calmar mínimo entre los 3 feeds (con pre/H1/H2 > 0):")
    ok = df[(df.pre_net > 0) & (df.h1_net > 0) & (df.h2_net > 0)]
    cols = keys + ["bitstamp_net", "bitstamp_dd", "bitstamp_n", "calmar_min", "pre_net", "h1_net", "h2_net"]
    print(ok.sort_values("calmar_min", ascending=False)[cols].head(15).round(2).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("fetch")
    ev = sub.add_parser("eval")
    for k, v in DEFAULTS.items():
        if isinstance(v, bool):
            ev.add_argument(f"--{k}", type=lambda x: x.lower() in ("1", "true", "on", "si", "sí"), default=v)
        else:
            ev.add_argument(f"--{k}", type=type(v), default=v)
    ev.add_argument("--trades", action="store_true", help="imprimir lista de operaciones")
    op = sub.add_parser("optimize")
    op.add_argument("--out", default="macd_sma800_6h_grid.csv")
    a = ap.parse_args()
    if a.cmd == "fetch":
        fetch_all()
    elif a.cmd == "optimize":
        optimize(a.out)
    else:
        p = {k: getattr(a, k) for k in DEFAULTS}
        for s in SOURCES:
            st, tr = Backtester(s).run(p)
            print(f"{s:9s} neto {st['net']:8.1f}%  CAGR {st['cagr']:5.1f}%  maxDD {st['dd']:5.1f}%  "
                  f"ops {st['trades']:3d}  ganadoras {st['wins']:3d}  PF {st['pf']:.2f}")
            if a.trades:
                print(tr.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
