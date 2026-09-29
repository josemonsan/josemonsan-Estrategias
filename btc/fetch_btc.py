"""
fetch_btc.py
=======================================================================
Descarga velas 6H de BTCUSDT (Binance spot) desde data-api.binance.vision
(espejo público de solo lectura; api.binance.com bloquea algunas regiones).

Uso:
    python3 btc/fetch_btc.py [--symbol BTCUSDT] [--interval 6h]
                             [--start 2017-08-01] [--out data/btcusdt_6h.csv]

Las velas se guardan en UTC con el tiempo de APERTURA, igual que TradingView
(BINANCE:BTCUSDT, 6H: 00/06/12/18 UTC).
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

BASE = "https://data-api.binance.vision/api/v3/klines"


def fetch_klines(symbol: str, interval: str, start: str) -> pd.DataFrame:
    start_ms = int(datetime.fromisoformat(start).replace(tzinfo=timezone.utc).timestamp() * 1000)
    rows = []
    while True:
        url = f"{BASE}?symbol={symbol}&interval={interval}&startTime={start_ms}&limit=1000"
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    batch = json.loads(r.read())
                break
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2 ** (attempt + 1))
        if not batch:
            break
        rows.extend(batch)
        start_ms = batch[-1][0] + 1
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=["t", "open", "high", "low", "close", "volume",
                                     "ct", "qv", "n", "tb", "tq", "ig"])
    df["time"] = pd.to_datetime(df["t"], unit="ms", utc=True)
    df = df[["time", "open", "high", "low", "close", "volume"]].astype(
        {c: float for c in ["open", "high", "low", "close", "volume"]})
    # La última vela puede estar abierta: descartarla.
    now = pd.Timestamp.now(tz="UTC")
    step = pd.Timedelta(interval)
    df = df[df["time"] + step <= now]
    return df.drop_duplicates("time").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--interval", default="6h")
    ap.add_argument("--start", default="2017-08-01")
    ap.add_argument("--out", default="data/btcusdt_6h.csv")
    a = ap.parse_args()
    df = fetch_klines(a.symbol, a.interval, a.start)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} velas {df['time'].iloc[0]} → {df['time'].iloc[-1]} → {a.out}")


if __name__ == "__main__":
    main()
