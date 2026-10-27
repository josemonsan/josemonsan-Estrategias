"""
Motor de backtest: tendencia SMA + Donchian, piramidación tipo Tortugas y
salida escalonada con alerta on-chain.

Reglas (todas las señales se calculan al CIERRE de la barra; las órdenes de
mercado se ejecutan en la APERTURA siguiente; stops y añadidos son órdenes
stop dentro de la barra):

Entrada   largo: cierre > máximo de las `donchian_entry` barras previas y cierre > SMA.
          corto: espejo (si allow_shorts). Sin entrada larga si hay alerta on-chain.
Tamaño    unidad = risk_per_unit * equity / (stop_atr * N). Exposición total <= max_leverage.
Piramida  +1 unidad cada `add_step_atr` * N de avance desde la última entrada, hasta
          `max_units`. Nunca con alerta on-chain activa.
Stop      común a `stop_atr` * N de la última entrada (solo se mueve a favor).
Salida    Donchian de `donchian_exit` barras; con alerta on-chain, la mitad.
On-chain  (solo largos) al saltar la alerta: cierra todas las unidades menos la primera
          y ciñe el stop a `stop_atr_alert` * N. Si la alerta se confirma: sale de todo.

Hipótesis conservadoras: si en la misma barra se tocan stop y añadido, primero el stop;
si tras un añadido el mínimo de la barra alcanza el nuevo stop, se asume que saltó.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import CostParams, StrategyParams


def wilder_atr(h, l, c, n):
    pc = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - pc), np.abs(l - pc)]), axis=0)
    atr = np.full_like(c, np.nan)
    if len(c) <= n:
        return atr
    atr[n] = np.nanmean(tr[1:n + 1])
    for i in range(n + 1, len(c)):
        atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / n
    return atr


@dataclass
class Unit:
    qty: float
    price: float
    time: pd.Timestamp


@dataclass
class BTResult:
    equity: pd.Series
    trades: pd.DataFrame
    fills: pd.DataFrame
    exposure: pd.Series
    params: StrategyParams
    extra: dict = field(default_factory=dict)


def run_backtest(df: pd.DataFrame, p: StrategyParams, c: CostParams,
                 initial_equity: float = 10_000.0) -> BTResult:
    idx = df.index
    o, h, l, cl = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    n = len(df)
    bar_hours = pd.Series(idx).diff().median() / pd.Timedelta(hours=1)

    atr = wilder_atr(h, l, cl, p.atr_len)
    s_h, s_l, s_c = pd.Series(h), pd.Series(l), pd.Series(cl)
    sma = s_c.rolling(p.sma_regime).mean().to_numpy()
    hh_in = s_h.rolling(p.donchian_entry).max().shift(1).to_numpy()
    ll_in = s_l.rolling(p.donchian_entry).min().shift(1).to_numpy()
    ex_n, ex_a = p.donchian_exit, max(2, p.donchian_exit // 2)
    ll_ex = s_l.rolling(ex_n).min().shift(1).to_numpy()
    ll_ex_a = s_l.rolling(ex_a).min().shift(1).to_numpy()
    hh_ex = s_h.rolling(ex_n).max().shift(1).to_numpy()

    use_oc = p.use_onchain and "oc_alert" in df
    alert = df["oc_alert"].to_numpy(bool) if use_oc else np.zeros(n, bool)
    confirm = df["oc_confirm"].to_numpy(bool) if use_oc else np.zeros(n, bool)

    const_f = c.funding_rate_8h * bar_hours / 8.0
    if "funding" in df:
        fund = df["funding"].fillna(const_f).to_numpy(float)
    else:
        fund = np.full(n, const_f)

    warm = max(p.sma_regime, p.donchian_entry, p.donchian_exit, p.atr_len) + 1

    cash = initial_equity
    side = 0
    units: list[Unit] = []
    stop = np.nan
    n_last = np.nan
    last_fill = np.nan
    alert_applied = False
    pending: dict | None = None
    camp: dict | None = None

    equity = np.full(n, np.nan)
    expo = np.zeros(n)
    trades, fills = [], []

    def qty_total():
        return sum(u.qty for u in units)

    def mark(px):
        return cash + sum(side * u.qty * (px - u.price) for u in units)

    def fill(i, px, qty, sd, kind):
        """Ejecuta una operación; sd = +1 compra, -1 venta."""
        nonlocal cash
        px_eff = px * (1 + sd * c.slippage)
        fee = abs(qty) * px_eff * c.fee_rate
        cash -= fee
        if camp is not None:
            camp["fees"] += fee
        fills.append({"time": idx[i], "kind": kind, "side": sd, "qty": qty,
                      "price": px_eff, "fee": fee})
        return px_eff

    def close_units(i, px, which, kind):
        nonlocal cash, side, units, stop, camp, alert_applied
        for u in which:
            pe = fill(i, px, u.qty, -side, kind)
            pnl = side * u.qty * (pe - u.price)
            cash += pnl
            camp["pnl_gross"] += pnl
        units = [u for u in units if all(u is not w for w in which)]
        if not units:
            camp.update(exit_time=idx[i], exit_price=px, exit_reason=kind)
            camp["pnl"] = camp["pnl_gross"] - camp["fees"] - camp["funding"]
            camp["R"] = camp["pnl"] / camp["risk0"] if camp["risk0"] > 0 else np.nan
            trades.append(camp)
            camp = None
            side = 0
            stop = np.nan
            alert_applied = False

    def open_unit(i, px, sd, n_atr, eq_ref, kind):
        nonlocal side, stop, n_last, last_fill, camp
        if not np.isfinite(n_atr) or n_atr <= 0:
            return False
        qty = p.risk_per_unit * eq_ref / (p.stop_atr * n_atr)
        cap = p.max_leverage * eq_ref / px - qty_total()
        qty = min(qty, cap)
        if qty <= 0:
            return False
        if side == 0:
            side = sd
            camp = {"side": sd, "entry_time": idx[i], "entry_price": px,
                    "risk0": qty * p.stop_atr * n_atr, "units_max": 0,
                    "pnl_gross": 0.0, "fees": 0.0, "funding": 0.0}
        pe = fill(i, px, qty, sd, kind)
        units.append(Unit(qty, pe, idx[i]))
        camp["units_max"] = max(camp["units_max"], len(units))
        n_last, last_fill = n_atr, pe
        new_stop = pe - sd * p.stop_atr * n_atr
        stop = new_stop if not np.isfinite(stop) else (
            max(stop, new_stop) if sd == 1 else min(stop, new_stop))
        return True

    for i in range(n):
        # 1) órdenes de mercado pendientes, a la apertura
        if pending is not None:
            act, pending = pending, None
            if act["type"] == "exit_all" and side != 0:
                close_units(i, o[i], list(units), act["reason"])
            elif act["type"] == "trim" and side != 0 and len(units) > 1:
                close_units(i, o[i], units[1:], "onchain_trim")
            elif act["type"] == "enter" and side == 0:
                open_unit(i, o[i], act["side"], atr[i - 1], equity[i - 1], "entry")

        # 2) stop dentro de la barra
        if side == 1 and l[i] <= stop:
            close_units(i, min(o[i], stop), list(units), "stop")
        elif side == -1 and h[i] >= stop:
            close_units(i, max(o[i], stop), list(units), "stop")

        # 3) piramidación (órdenes stop), sin alerta conocida al cierre previo
        if side != 0 and i > 0 and not alert[i - 1]:
            while side != 0 and len(units) < p.max_units:
                trig = last_fill + side * p.add_step_atr * n_last
                hit = h[i] >= trig if side == 1 else l[i] <= trig
                if not hit:
                    break
                px = max(o[i], trig) if side == 1 else min(o[i], trig)
                if not open_unit(i, px, side, atr[i - 1], mark(px), "add"):
                    break
                if (side == 1 and l[i] <= stop) or (side == -1 and h[i] >= stop):
                    close_units(i, stop, list(units), "stop_same_bar")

        # 4) funding (largos pagan si la tasa es positiva)
        if side != 0:
            f = side * qty_total() * cl[i] * fund[i]
            cash -= f
            camp["funding"] += f

        # 5) valoración
        equity[i] = mark(cl[i])
        expo[i] = side * qty_total() * cl[i] / equity[i] if equity[i] > 0 else 0.0
        if equity[i] <= 0:
            equity[i:] = 0.0
            break

        # 6) señales al cierre para la barra siguiente
        if i < warm or i == n - 1 or not np.isfinite(atr[i]):
            continue
        if side == 0:
            if cl[i] > hh_in[i] and cl[i] > sma[i] and not alert[i]:
                pending = {"type": "enter", "side": 1}
            elif p.allow_shorts and cl[i] < ll_in[i] and cl[i] < sma[i]:
                pending = {"type": "enter", "side": -1}
        elif side == 1:
            if use_oc and confirm[i]:
                pending = {"type": "exit_all", "reason": "onchain_confirm"}
            elif cl[i] < (ll_ex_a[i] if alert[i] else ll_ex[i]):
                pending = {"type": "exit_all", "reason": "donchian_exit"}
            elif use_oc and alert[i] and not alert_applied:
                stop = max(stop, cl[i] - p.stop_atr_alert * atr[i])
                if len(units) > 1:
                    pending = {"type": "trim"}
                alert_applied = True
            if not alert[i]:
                alert_applied = False
        elif side == -1:
            if cl[i] > hh_ex[i]:
                pending = {"type": "exit_all", "reason": "donchian_exit"}

    # cierre contable de la posición abierta al final (sin costes) para las estadísticas
    if side != 0 and camp is not None:
        last = n - 1
        open_pnl = sum(side * u.qty * (cl[last] - u.price) for u in units)
        camp.update(exit_time=idx[last], exit_price=cl[last], exit_reason="open_at_end",
                    pnl=camp["pnl_gross"] + open_pnl - camp["fees"] - camp["funding"])
        camp["R"] = camp["pnl"] / camp["risk0"]
        trades.append(camp)

    eq = pd.Series(equity, index=idx).ffill().fillna(initial_equity)
    tr = pd.DataFrame(trades)
    if not tr.empty:
        tr = tr.drop(columns=["pnl_gross"])
    return BTResult(eq, tr, pd.DataFrame(fills), pd.Series(expo, index=idx), p)
