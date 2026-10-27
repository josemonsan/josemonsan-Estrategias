"""Carga y descarga de precios y funding. Todo en UTC."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

TF_MS = {"1h": 3_600_000, "4h": 14_400_000, "6h": 21_600_000, "1d": 86_400_000}


def fetch_ohlcv_ccxt(exchange: str = "okx", symbol: str | None = None,
                     timeframe: str = "1d", since: str = "2017-01-01") -> pd.DataFrame:
    """Descarga OHLCV paginado. okx -> perpetuo ETH/USDT:USDT (historia desde ~2019-12);
    binance -> spot ETH/USDT (historia desde 2017-08)."""
    import ccxt

    ex = getattr(ccxt, exchange)({"enableRateLimit": True})
    if symbol is None:
        symbol = "ETH/USDT:USDT" if exchange == "okx" else "ETH/USDT"
    step = TF_MS[timeframe]
    since_ms = ex.parse8601(f"{since}T00:00:00Z")
    now_ms = ex.milliseconds()
    rows: list[list] = []
    while since_ms < now_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since_ms, limit=300)
        if not batch:
            # OKX a veces no devuelve nada antes del listado: salta hacia delante
            since_ms += step * 300
            continue
        rows.extend(batch)
        nxt = batch[-1][0] + step
        if nxt <= since_ms:
            break
        since_ms = nxt
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("time").sort_values("time")
    df = df[df["time"] + step <= now_ms]          # quita la vela en curso
    df.index = pd.to_datetime(df.pop("time"), unit="ms", utc=True)
    return df.astype(float)


def load_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Acepta el CSV de este script o una exportación de TradingView
    (columna time en segundos unix, o fecha ISO)."""
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    tcol = next(c for c in ("time", "timestamp", "date", "datetime") if c in df.columns)
    t = df.pop(tcol)
    if pd.api.types.is_numeric_dtype(t):
        unit = "ms" if t.iloc[0] > 1e11 else "s"
        idx = pd.to_datetime(t, unit=unit, utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    df.index = idx
    df = df[["open", "high", "low", "close"] + (["volume"] if "volume" in df else [])]
    return df.sort_index().astype(float)


def load_funding_csv(path: str | Path) -> pd.Series:
    """CSV con columnas time, rate (tasa por evento de funding, p. ej. cada 8h)."""
    f = pd.read_csv(path)
    f.columns = [c.strip().lower() for c in f.columns]
    t = f["time"]
    if pd.api.types.is_numeric_dtype(t):
        idx = pd.to_datetime(t, unit="ms" if t.iloc[0] > 1e11 else "s", utc=True)
    else:
        idx = pd.to_datetime(t, utc=True)
    return pd.Series(f["rate"].astype(float).values, index=idx).sort_index()


def bar_delta(index: pd.DatetimeIndex) -> pd.Timedelta:
    return pd.Series(index).diff().median()


def funding_per_bar(funding: pd.Series, bars: pd.DatetimeIndex) -> pd.Series:
    """Suma los eventos de funding que caen dentro de cada barra [open, close)."""
    pos = bars.searchsorted(funding.index, side="right") - 1
    ok = pos >= 0
    out = pd.Series(0.0, index=bars)
    np.add.at(out.values, pos[ok], funding.values[ok])
    first = funding.index.min()
    out[bars < first.floor("D")] = np.nan     # sin dato antes del primer funding
    return out
