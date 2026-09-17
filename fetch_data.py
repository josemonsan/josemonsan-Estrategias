"""
fetch_data.py
=======================================================================
Utilidades de descarga de datos reales para respaldar los backtests de
cava_trend_strategy_v3.py sobre mercado real (ver notes/2026-09-17-*.md).
Consolida en un solo lugar lo que antes eran varios scripts ad hoc de
scratchpad (no reproducibles fuera de esa sesión): precios diarios
(Yahoo Finance), base monetaria USD mensual (FRED, BOGMBASE) y fechas de
publicación de resultados (Yahoo Finance).

Uso como librería:
    from fetch_data import fetch_prices, fetch_macro, fetch_earnings

Uso como script:
    python3 fetch_data.py [--universe universe_43.csv] [--start 2015-01-01]
                           [--out-dir data] [--skip-prices] [--skip-macro]
                           [--skip-earnings]

Los datos descargados NO se versionan (ver .gitignore); son
regenerables desde cero con este script. Requiere `yfinance`, no
declarado como dependencia del motor (cava_trend_strategy_v3.py solo
necesita numpy/pandas) -- instalar con `pip install yfinance`.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
import time
import urllib.request
from pathlib import Path

import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None  # solo se necesita para fetch_prices/fetch_earnings


def load_universe(path: str) -> list[str]:
    with open(path, newline="") as f:
        return [row["ticker"] for row in csv.DictReader(f)]


def fetch_prices(tickers: list[str], start: str = "2015-01-01",
                  retries: int = 4, retry_sleep_s: float = 4.0,
                  min_rows: int = 300) -> dict[str, pd.DataFrame]:
    """OHLCV diario ajustado (auto_adjust=True) por ticker, indexado por fecha tz-naive."""
    if yf is None:
        raise ImportError("fetch_prices requiere 'yfinance' (pip install yfinance)")
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        for attempt in range(retries):
            try:
                df = yf.download(t, start=start, auto_adjust=True, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                df = df.rename(columns=str.lower).dropna(how="any")
                df = df[["open", "high", "low", "close", "volume"]]
                df.index = pd.DatetimeIndex(df.index).tz_localize(None)
                if len(df) >= min_rows:
                    out[t] = df
                    print(f"  {t}: {len(df)} filas OK")
                else:
                    print(f"  [aviso] {t}: solo {len(df)} filas, descartado")
                break
            except Exception as e:
                print(f"  [retry {attempt}] {t}: {e}")
                time.sleep(retry_sleep_s)
        else:
            print(f"  [FAIL] {t}: sin datos tras {retries} intentos")
    return out


def fetch_macro(fred_series: str = "BOGMBASE", since: str = "2010-01-01") -> pd.DataFrame:
    """Serie mensual de FRED (por defecto base monetaria USD, la que lee macro_regime())."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={fred_series}"
    with urllib.request.urlopen(url, timeout=20) as resp:
        raw = resp.read()
    df = pd.read_csv(io.BytesIO(raw), parse_dates=["observation_date"])
    df = df.rename(columns={"observation_date": "date", fred_series: "monetary_base"})
    df = df.set_index("date").sort_index()
    df = df[df.index >= since]
    return df.resample("ME").last()


def fetch_earnings(tickers: list[str], limit: int = 80,
                    retries: int = 4, retry_sleep_s: float = 4.0) -> dict[str, pd.DatetimeIndex]:
    """Fechas de publicación de resultados por ticker (Yahoo Finance)."""
    if yf is None:
        raise ImportError("fetch_earnings requiere 'yfinance' (pip install yfinance)")
    out: dict[str, pd.DatetimeIndex] = {}
    for t in tickers:
        for attempt in range(retries):
            try:
                ed = yf.Ticker(t).get_earnings_dates(limit=limit)
                if ed is None or ed.empty:
                    print(f"  [aviso] {t}: sin earnings dates")
                    out[t] = pd.DatetimeIndex([])
                else:
                    idx = pd.DatetimeIndex(ed.index).tz_localize(None).normalize()
                    out[t] = idx
                    print(f"  {t}: {len(idx)} earnings dates, {idx.min().date()} -> {idx.max().date()}")
                break
            except Exception as e:
                print(f"  [retry {attempt}] {t}: {e}")
                time.sleep(retry_sleep_s)
        else:
            print(f"  [FAIL] {t}: sin datos tras {retries} intentos")
            out[t] = pd.DatetimeIndex([])
    return out


def _save_prices_csv(prices: dict[str, pd.DataFrame], path: Path) -> None:
    rows = []
    for t, df in prices.items():
        d = df.reset_index().rename(columns={"index": "date", "Date": "date"})
        d.insert(0, "ticker", t)
        rows.append(d)
    pd.concat(rows, ignore_index=True).to_csv(path, index=False)


def _save_macro_csv(macro: pd.DataFrame, path: Path) -> None:
    macro.reset_index().rename(columns={"index": "date"}).to_csv(path, index=False)


def _save_earnings_csv(earnings: dict[str, pd.DatetimeIndex], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ticker", "earnings_date"])
        for t, idx in earnings.items():
            for d in idx:
                w.writerow([t, d.date().isoformat()])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--universe", default="universe_43.csv", help="CSV con columna 'ticker'")
    ap.add_argument("--start", default="2015-01-01", help="Fecha de inicio para precios")
    ap.add_argument("--out-dir", default="data", help="Directorio de salida (no versionado)")
    ap.add_argument("--skip-prices", action="store_true")
    ap.add_argument("--skip-macro", action="store_true")
    ap.add_argument("--skip-earnings", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tickers = load_universe(args.universe)
    print(f"Universo: {len(tickers)} tickers desde {args.universe}")

    if not args.skip_prices:
        print("\nDescargando precios (Yahoo Finance)...")
        prices = fetch_prices(tickers, start=args.start)
        _save_prices_csv(prices, out_dir / "prices.csv")
        print(f"  {len(prices)} tickers OK -> {out_dir / 'prices.csv'}")

    if not args.skip_macro:
        print("\nDescargando base monetaria USD (FRED, BOGMBASE)...")
        macro = fetch_macro()
        _save_macro_csv(macro, out_dir / "macro_monetary_base.csv")
        print(f"  {len(macro)} observaciones mensuales -> {out_dir / 'macro_monetary_base.csv'}")

    if not args.skip_earnings:
        print("\nDescargando fechas de earnings (Yahoo Finance)...")
        earnings = fetch_earnings(tickers)
        _save_earnings_csv(earnings, out_dir / "earnings_dates.csv")
        n_dates = sum(len(v) for v in earnings.values())
        print(f"  {len(earnings)} tickers, {n_dates} fechas -> {out_dir / 'earnings_dates.csv'}")


if __name__ == "__main__":
    main()
