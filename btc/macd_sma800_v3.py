"""
macd_sma800_v3.py
=======================================================================
Port a Python de pine/macd_sma800_6h_v3.pine (MACD D1 régimen + SMA800 6H
entrada + Chandelier ATR salida) para poder barrer parámetros fuera de
TradingView (TradingView no expone API para lanzar el Strategy Tester).

Replica el emulador de brokers de TradingView en lo relevante:
  - La lógica se evalúa al CIERRE de cada vela; strategy.entry/close se
    ejecutan a la APERTURA de la vela siguiente (process_orders_on_close=false).
  - strategy.exit(stop=...) queda activa desde la vela siguiente; si la vela
    abre por debajo del stop (gap) se llena a la apertura. La vela en la que
    entra la posición ya puede tocar el stop.
  - Comisión % por lado, slippage en ticks, qty = % del equity / close de la
    vela de señal.
  - Indicadores con las mismas definiciones que Pine (RMA/EMA sembradas con
    SMA, ta.dmi, ta.stoch, ta.rsi).
  - D1: velas diarias UTC reconstruidas desde 6H; cada vela 6H ve el valor de
    la vela diaria CERRADA anterior (security(..., x[1], lookahead_on)).

Diferencias conocidas: el cierre parcial (P) no está portado (apagado por
defecto en la v3); el Max DD se mide sobre equity al cierre (igual que la
tabla del script, no el de la pestaña de resumen de TV, que es intrabar).
"""
from __future__ import annotations

from dataclasses import dataclass, replace, asdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "btcusdt_6h.csv"


# ---------------------------------------------------------------- Params
@dataclass(frozen=True)
class P:
    macdFast: int = 12
    macdSlow: int = 26
    exitOnRegime: bool = True
    holdTrend: bool = False
    dirMode: str = "Solo largos"   # "Ambas" | "Solo largos" | "Solo cortos"
    smaLen: int = 800
    useAdx: bool = True
    adxLen: int = 14
    adxThr: float = 20
    useSlope: bool = False
    slopeLen: int = 20
    useAtrD: bool = True
    useReentry: bool = True
    reLen: int = 20
    rsiLen: int = 14
    useRsiMom: bool = False
    rsiMin: float = 50
    useRsiCap: bool = False
    rsiMax: float = 75
    useStochPb: bool = False
    stLen: int = 14
    stSmooth: int = 3
    stOS: float = 20
    atrLen: int = 14
    slMult: float = 3.0
    chMult: float = 8.0
    # strategy() header
    capital: float = 10000.0
    pct: float = 100.0
    comm: float = 0.0035     # % por lado
    slipTicks: int = 1
    tick: float = 0.01


# ---------------------------------------------------------------- Pine TA
def sma(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).rolling(n, min_periods=n).mean().to_numpy()


def _seeded(x: np.ndarray, n: int, alpha: float) -> np.ndarray:
    out = np.full(len(x), np.nan)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < n:
        return out
    s = valid[0]
    if np.isnan(x[s:s + n]).any():
        return out
    prev = x[s:s + n].mean()
    out[s + n - 1] = prev
    for i in range(s + n, len(x)):
        prev = alpha * x[i] + (1 - alpha) * prev
        out[i] = prev
    return out


def rma(x, n):
    return _seeded(np.asarray(x, float), n, 1.0 / n)


def ema(x, n):
    return _seeded(np.asarray(x, float), n, 2.0 / (n + 1))


def true_range(h, l, c):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    tr[0] = h[0] - l[0]
    return tr


def rsi(c, n):
    ch = np.r_[np.nan, np.diff(c)]
    up = rma(np.where(np.isnan(ch), np.nan, np.maximum(ch, 0)), n)
    dn = rma(np.where(np.isnan(ch), np.nan, np.maximum(-ch, 0)), n)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(dn == 0, 100.0, np.where(up == 0, 0.0, 100 - 100 / (1 + up / dn)))
    r[np.isnan(up) | np.isnan(dn)] = np.nan
    return r


def stoch(c, h, l, n):
    hh = pd.Series(h).rolling(n, min_periods=n).max().to_numpy()
    ll = pd.Series(l).rolling(n, min_periods=n).min().to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100 * (c - ll) / (hh - ll)


def adx(h, l, c, n):
    up = np.r_[np.nan, np.diff(h)]
    dn = np.r_[np.nan, -np.diff(l)]
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    mdm = np.where((dn > up) & (dn > 0), dn, 0.0)
    pdm[0] = mdm[0] = np.nan
    tr = true_range(h, l, c)
    tr[0] = np.nan
    trur = rma(tr, n)
    with np.errstate(divide="ignore", invalid="ignore"):
        plus = 100 * rma(pdm, n) / trur
        minus = 100 * rma(mdm, n) / trur
        s = plus + minus
        return 100 * rma(np.abs(plus - minus) / np.where(s == 0, 1, s), n)


def shift(x, k):
    out = np.full(len(x), np.nan)
    if k < len(x):
        out[k:] = x[:len(x) - k]
    return out


# ---------------------------------------------------------------- Data
class Market:
    def __init__(self, path: Path = DATA, start: str | None = None, end: str | None = None):
        df = pd.read_csv(path, parse_dates=["time"])
        if start:
            df = df[df.time >= pd.Timestamp(start, tz="UTC")]
        if end:
            df = df[df.time < pd.Timestamp(end, tz="UTC")]
        self.df = df.reset_index(drop=True)
        self.t = self.df.time
        self.o, self.h, self.l, self.c = (self.df[k].to_numpy(float) for k in ["open", "high", "low", "close"])
        d = self.df.set_index("time").resample("1D").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
        self.d = d
        # índice de la vela diaria a la que pertenece cada vela 6H
        day = self.t.dt.floor("1D")
        self.dayidx = d.index.get_indexer(day)
        self._cache: dict = {}

    def _daily_prev(self, arr_d: np.ndarray) -> np.ndarray:
        """Valor de la vela diaria CERRADA anterior, alineado a 6H."""
        prev = np.r_[np.nan, arr_d[:-1]]
        return prev[self.dayidx]

    def get(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    def macdD(self, f, s):
        dc = self.d.close.to_numpy()
        return self.get(("macdD", f, s), lambda: self._daily_prev(ema(dc, f) - ema(dc, s)))

    def atrD(self, n):
        d = self.d
        return self.get(("atrD", n), lambda: self._daily_prev(
            rma(true_range(d.high.to_numpy(), d.low.to_numpy(), d.close.to_numpy()), n)))

    def atr6(self, n):
        return self.get(("atr6", n), lambda: rma(true_range(self.h, self.l, self.c), n))

    def sma(self, n):
        return self.get(("sma", n), lambda: sma(self.c, n))

    def adx(self, n):
        return self.get(("adx", n), lambda: adx(self.h, self.l, self.c, n))

    def rsi(self, n):
        return self.get(("rsi", n), lambda: rsi(self.c, n))

    def stK(self, n, sm):
        return self.get(("stK", n, sm), lambda: sma(stoch(self.c, self.h, self.l, n), sm))

    def hi(self, n):
        return self.get(("hi", n), lambda: shift(pd.Series(self.h).rolling(n, min_periods=n).max().to_numpy(), 1))

    def lo(self, n):
        return self.get(("lo", n), lambda: shift(pd.Series(self.l).rolling(n, min_periods=n).min().to_numpy(), 1))


def _gt(a, b):
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(a, nan=np.nan) > b


def signals(m: Market, p: P) -> dict:
    """Series booleanas de la lógica Pine (todo vectorizable)."""
    c = m.c
    s = m.sma(p.smaLen)
    sPrev = shift(s, 1)
    cPrev = shift(c, 1)
    sBack = shift(s, p.slopeLen)
    macd = m.macdD(p.macdFast, p.macdSlow)
    with np.errstate(invalid="ignore"):
        regL = macd > 0
        regS = macd < 0
        adxOK = np.ones(len(c), bool) if not p.useAdx else (m.adx(p.adxLen) > p.adxThr)
        slopeL = np.ones(len(c), bool) if not p.useSlope else (s > sBack)
        slopeS = np.ones(len(c), bool) if not p.useSlope else (s < sBack)
        trendUp = (c > s) & (s > sBack)
        trendDn = (c < s) & (s < sBack)
        crossUp = (c > s) & (cPrev <= sPrev)
        crossDn = (c < s) & (cPrev >= sPrev)
        hiBrk = c > m.hi(p.reLen)
        loBrk = c < m.lo(p.reLen)
        r = m.rsi(p.rsiLen)
        rsiOKL = np.ones(len(c), bool)
        rsiOKS = np.ones(len(c), bool)
        if p.useRsiMom:
            rsiOKL &= r >= p.rsiMin
            rsiOKS &= r <= 100 - p.rsiMin
        if p.useRsiCap:
            rsiOKL &= r <= p.rsiMax
            rsiOKS &= r >= 100 - p.rsiMax
        rawBrkL = p.useReentry & (c > s) & hiBrk
        rawBrkS = p.useReentry & (c < s) & loBrk
        okCrossL, okCrossS = crossUp & rsiOKL, crossDn & rsiOKS
        okBrkL, okBrkS = rawBrkL & rsiOKL, rawBrkS & rsiOKS
        if p.useStochPb:
            k = m.stK(p.stLen, p.stSmooth)
            kp = shift(k, 1)
            okPbL = (c > s) & (k > p.stOS) & (kp <= p.stOS)
            okPbS = (c < s) & (k < 100 - p.stOS) & (kp >= 100 - p.stOS)
        else:
            okPbL = okPbS = np.zeros(len(c), bool)
    return dict(
        regL=regL, regS=regS, adxOK=adxOK, slopeL=slopeL, slopeS=slopeS,
        trendUp=trendUp, trendDn=trendDn,
        okCrossL=okCrossL, okCrossS=okCrossS, okBrkL=okBrkL, okBrkS=okBrkS,
        okPbL=okPbL, okPbS=okPbS,
        atr=m.atrD(p.atrLen) if p.useAtrD else m.atr6(p.atrLen),
    )


# ---------------------------------------------------------------- Broker
def run(m: Market, p: P) -> dict:
    sg = signals(m, p)
    o, h, l, c = m.o, m.h, m.l, m.c
    n = len(c)
    slip = p.slipTicks * p.tick
    fee = p.comm / 100
    allowL = p.dirMode != "Solo cortos"
    allowS = p.dirMode != "Solo largos"

    L = {k: sg[k].tolist() for k in sg if k != "atr"}
    atr = sg["atr"].tolist()
    ol, hl, ll_, cl = o.tolist(), h.tolist(), l.tolist(), c.tolist()

    cash = p.capital
    qty = 0.0            # >0 largo, <0 corto
    avg = 0.0
    entry_type = ""
    entry_bar = -1
    pend = None          # ("L"/"S"/"close", qty_signal, type)
    stopL = stopS = None  # stop activo para la vela siguiente
    riskL = riskS = trailL = trailS = hh = lo_ = float("nan")
    prev_qty = 0.0
    trades = []
    equity = np.empty(n)

    def fill(side_qty, price, bar, tag):
        nonlocal cash, qty, avg, entry_type, entry_bar
        # cerrar posición existente si cambia de signo o es cierre
        if qty != 0 and (side_qty == 0 or np.sign(side_qty) != np.sign(qty)):
            pnl = qty * (price - avg)
            f = abs(qty) * price * fee
            cash += pnl - f
            trades.append(dict(side="L" if qty > 0 else "S", type=entry_type, entry_bar=entry_bar,
                               exit_bar=bar, entry=avg, exit=price, qty=qty, pnl=pnl - f - trades_fee.pop(),
                               reason=tag))
            qty = 0.0
        if side_qty != 0 and qty == 0:
            f = abs(side_qty) * price * fee
            trades_fee.append(f)
            qty, avg, entry_bar = side_qty, price, bar

    trades_fee: list = []

    for i in range(n):
        # ---- 1) apertura: órdenes de mercado pendientes
        if pend is not None:
            kind, sq, typ = pend
            if kind == "close":
                if qty != 0:
                    fill(0.0, ol[i] - slip if qty > 0 else ol[i] + slip, i, typ)
            elif kind == "L":
                fill(sq, ol[i] + slip, i, "Reversal")
                entry_type = typ
            else:
                fill(-sq, ol[i] - slip, i, "Reversal")
                entry_type = typ
            pend = None
        # ---- 2) intrabar: stops
        if qty > 0 and stopL is not None and ll_[i] <= stopL:
            px = min(ol[i], stopL) - slip
            fill(0.0, px, i, "Trail")
        elif qty < 0 and stopS is not None and hl[i] >= stopS:
            px = max(ol[i], stopS) + slip
            fill(0.0, px, i, "Trail")
        # ---- 3) cierre: lógica del script
        eq = cash + (qty * (cl[i] - avg) if qty != 0 else 0.0)
        equity[i] = eq
        flatOrS = qty <= 0
        flatOrL = qty >= 0
        trigL = L["okCrossL"][i] or L["okBrkL"][i] or L["okPbL"][i]
        trigS = L["okCrossS"][i] or L["okBrkS"][i] or L["okPbS"][i]
        entryL = trigL and L["regL"][i] and allowL and flatOrS and L["slopeL"][i] and L["adxOK"][i]
        entryS = trigS and L["regS"][i] and allowS and flatOrL and L["slopeS"][i] and L["adxOK"][i]
        a = atr[i]
        newStopL = newStopS = None
        if entryL:
            typ = "cruce" if L["okCrossL"][i] else "ruptura" if L["okBrkL"][i] else "pullback"
            sq = eq * p.pct / 100 / cl[i]
            pend = ("L", sq, typ)
            riskL = p.slMult * a
            trailL = cl[i] - riskL
            hh = cl[i]
        if entryS:
            typ = "cruce" if L["okCrossS"][i] else "ruptura" if L["okBrkS"][i] else "pullback"
            sq = eq * p.pct / 100 / cl[i]
            pend = ("S", sq, typ)
            riskS = p.slMult * a
            trailS = cl[i] + riskS
            lo_ = cl[i]
        posL, posS = qty > 0, qty < 0
        if posL:
            if prev_qty <= 0:
                trailL = avg - riskL
                hh = hl[i]
            else:
                hh = max(hh, hl[i])
            trailL = max(trailL, hh - p.chMult * a)
        if posS:
            if prev_qty >= 0:
                trailS = avg + riskS
                lo_ = ll_[i]
            else:
                lo_ = min(lo_, ll_[i])
            trailS = min(trailS, lo_ + p.chMult * a)
        if entryL or posL:
            newStopL = trailL
        if entryS or posS:
            newStopS = trailS
        stopL, stopS = newStopL, newStopS
        if p.exitOnRegime and posL and not L["regL"][i] and not (p.holdTrend and L["trendUp"][i]) and not entryS:
            pend = ("close", 0, "Régimen off")
        if p.exitOnRegime and posS and not L["regS"][i] and not (p.holdTrend and L["trendDn"][i]) and not entryL:
            pend = ("close", 0, "Régimen off")
        prev_qty = qty

    return dict(equity=equity, trades=pd.DataFrame(trades), open_qty=qty)


# ---------------------------------------------------------------- Stats
def stats(m: Market, res: dict, start: str | None = None, end: str | None = None) -> dict:
    """Métricas sobre la ventana [start, end) usando la equity de la corrida completa."""
    eq = pd.Series(res["equity"], index=m.t)
    mask = np.ones(len(eq), bool)
    if start:
        mask &= (eq.index >= pd.Timestamp(start, tz="UTC"))
    if end:
        mask &= (eq.index < pd.Timestamp(end, tz="UTC"))
    e = eq[mask]
    if len(e) < 2:
        return {}
    idx = np.where(mask)[0]
    base = eq.iloc[idx[0] - 1] if idx[0] > 0 else e.iloc[0]
    d = e.resample("1D").last().dropna()
    d = pd.concat([pd.Series([base], index=[d.index[0] - pd.Timedelta("1D")]), d])
    r = d.pct_change().dropna()
    yrs = (e.index[-1] - e.index[0]).days / 365.25
    net = e.iloc[-1] / base - 1
    cagr = (1 + net) ** (1 / yrs) - 1 if yrs > 0 and net > -1 else np.nan
    peak = np.maximum.accumulate(np.r_[base, e.to_numpy()])
    dd = ((peak - np.r_[base, e.to_numpy()]) / peak).max()
    sharpe = r.mean() / r.std() * np.sqrt(365) if r.std() > 0 else 0.0
    tr = res["trades"]
    if len(tr):
        tt = pd.DatetimeIndex(m.t.iloc[tr.exit_bar])
        sel = np.ones(len(tr), bool)
        if start:
            sel &= tt >= pd.Timestamp(start, tz="UTC")
        if end:
            sel &= tt < pd.Timestamp(end, tz="UTC")
        tr = tr[sel]
    gp = tr.pnl[tr.pnl > 0].sum() if len(tr) else 0
    gl = -tr.pnl[tr.pnl < 0].sum() if len(tr) else 0
    return dict(net=net * 100, cagr=cagr * 100, maxdd=dd * 100, sharpe=sharpe,
                mar=(cagr / dd) if dd > 0 else np.nan, trades=len(tr),
                win=(tr.pnl > 0).mean() * 100 if len(tr) else np.nan,
                pf=gp / gl if gl > 0 else np.nan)


if __name__ == "__main__":
    m = Market()
    res = run(m, P())
    print("Baseline v3 (defaults == v2):", {k: round(v, 3) for k, v in stats(m, res).items()})
    tr = res["trades"]
    tr["entry_t"] = m.t.iloc[tr.entry_bar].dt.strftime("%Y-%m-%d %H").values
    tr["exit_t"] = m.t.iloc[tr.exit_bar].dt.strftime("%Y-%m-%d %H").values
    print(tr[["side", "type", "entry_t", "exit_t", "entry", "exit", "pnl", "reason"]].round(2).to_string())
