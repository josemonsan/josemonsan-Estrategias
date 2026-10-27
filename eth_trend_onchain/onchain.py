"""
Métricas on-chain de ETH desde el dataset público de Google BigQuery
(bigquery-public-data.crypto_ethereum) y construcción de la alerta.

Métricas diarias (sin etiquetas salvo la lista corta de exchanges):
  whale_count / whale_volume : transferencias >= 1.000 ETH
  dormant_eth                : ETH enviado por direcciones que llevaban >= 365 días sin enviar
  vdd                        : value-days destroyed = sum(ETH x días inactivo, tope 10 años).
                               Es el análogo de coin days destroyed en un modelo de cuentas:
                               aproximación, no el CDD exacto de una cadena UTXO.
  exch_inflow / outflow / netflow : solo si pasas un CSV de direcciones de exchanges

Limitaciones: usa la tabla `transactions` (transferencias de ETH de primer nivel).
No incluye transferencias internas de contratos ni WETH/ERC-20.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import OnchainParams

SQL = r"""
DECLARE start_ts TIMESTAMP DEFAULT TIMESTAMP(@start);

WITH tx AS (
  SELECT
    block_timestamp AS ts,
    LOWER(from_address) AS from_address,
    LOWER(to_address)   AS to_address,
    SAFE_CAST(value AS FLOAT64) / 1e18 AS eth
  FROM `bigquery-public-data.crypto_ethereum.transactions`
  WHERE value > 0
    AND (receipt_status = 1 OR receipt_status IS NULL)   -- NULL antes de Byzantium (2017-10)
),
gaps AS (
  SELECT
    ts, eth, from_address, to_address,
    TIMESTAMP_DIFF(ts, LAG(ts) OVER (PARTITION BY from_address ORDER BY ts), SECOND)
      / 86400.0 AS idle_days
  FROM tx
)
SELECT
  DATE(ts) AS date,
  COUNT(*)                                        AS tx_count,
  SUM(eth)                                        AS total_volume,
  COUNTIF(eth >= 1000)                            AS whale_count,
  SUM(IF(eth >= 1000, eth, 0))                    AS whale_volume,
  SUM(IF(idle_days >= 365, eth, 0))               AS dormant_eth,
  SUM(eth * LEAST(IFNULL(idle_days, 0), 3650))    AS vdd,
  SUM(IF(to_address   IN UNNEST(@exch), eth, 0))  AS exch_inflow,
  SUM(IF(from_address IN UNNEST(@exch), eth, 0))  AS exch_outflow
FROM gaps
WHERE ts >= start_ts
GROUP BY date
ORDER BY date
"""
# Nota: el LAG se calcula sobre toda la historia (no solo desde @start) para que
# la inactividad de las primeras fechas sea correcta. Eso escanea la tabla entera.


def load_exchange_addresses(path: str | Path | None) -> list[str]:
    if not path:
        return []
    df = pd.read_csv(path, comment="#")
    return sorted({a.strip().lower() for a in df["address"].dropna() if a.strip()})


def fetch_onchain_bigquery(project: str, start: str = "2016-01-01",
                           exchanges_csv: str | None = None,
                           max_gb: float = 600.0, yes: bool = False) -> pd.DataFrame:
    """Lanza la consulta. Antes hace un dry-run e informa de los GB a escanear
    (plan gratuito: 1 TB/mes). Ejecútala una vez y guarda el CSV."""
    from google.cloud import bigquery

    exch = load_exchange_addresses(exchanges_csv)
    client = bigquery.Client(project=project)
    params = [
        bigquery.ScalarQueryParameter("start", "STRING", start),
        bigquery.ArrayQueryParameter("exch", "STRING", exch),
    ]
    dry = client.query(SQL, job_config=bigquery.QueryJobConfig(
        dry_run=True, use_query_cache=False, query_parameters=params))
    gb = dry.total_bytes_processed / 1e9
    print(f"[bigquery] dry-run: {gb:,.1f} GB a escanear "
          f"({len(exch)} direcciones de exchange)")
    if gb > max_gb:
        raise SystemExit(f"Supera --max-gb={max_gb}. Sube el límite si quieres seguir.")
    if not yes:
        if input("¿Ejecutar la consulta? [s/N] ").strip().lower() not in ("s", "si", "sí", "y"):
            raise SystemExit("Cancelado.")
    job = client.query(SQL, job_config=bigquery.QueryJobConfig(
        query_parameters=params,
        maximum_bytes_billed=int(max_gb * 1e9)))
    df = job.result().to_dataframe()
    if not exch:
        df[["exch_inflow", "exch_outflow"]] = np.nan
    return df


def load_onchain_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.index = pd.DatetimeIndex(pd.to_datetime(df.pop("date"), utc=True))
    if {"exch_inflow", "exch_outflow"} <= set(df.columns):
        df["exch_netflow"] = df["exch_inflow"] - df["exch_outflow"]
    return df.sort_index()


def build_signal(daily: pd.DataFrame, op: OnchainParams) -> pd.DataFrame:
    """Compuesto de riesgo on-chain = media de z-scores causales.
    Valores altos = más ETH grande/antiguo moviéndose o entrando a exchanges."""
    zs = {}
    for c in op.components:
        if c not in daily or daily[c].isna().all():
            continue
        x = daily[c].astype(float).rolling(op.window_days, min_periods=op.window_days).sum()
        if c != "exch_netflow":
            x = np.log1p(x.clip(lower=0))
        mu = x.rolling(op.z_lookback_days, min_periods=op.z_min_periods).mean()
        sd = x.rolling(op.z_lookback_days, min_periods=op.z_min_periods).std()
        zs[c] = (x - mu) / sd
    if not zs:
        raise ValueError("No hay componentes on-chain válidos en el CSV.")
    z = pd.DataFrame(zs)
    out = z.add_prefix("z_")
    out["composite"] = z.mean(axis=1, skipna=True)
    out["alert"] = out["composite"] > op.alert_z
    streak = out["alert"].astype(int).groupby((~out["alert"]).cumsum()).cumsum()
    out["confirm"] = streak >= op.confirm_days
    return out


def align_to_bars(sig: pd.DataFrame, bars: pd.DatetimeIndex, bar_td: pd.Timedelta,
                  lag_hours: int) -> pd.DataFrame:
    """Sin anticipación: el dato del día D (00:00-24:00 UTC) solo está disponible
    desde D+1 + lag_hours, y se usa en la primera barra que CIERRA después."""
    avail = sig.copy()
    avail.index = avail.index + pd.Timedelta(days=1) + pd.Timedelta(hours=lag_hours)
    left = pd.DataFrame({"bar_close": bars + bar_td, "bar_open": bars})
    m = pd.merge_asof(left.sort_values("bar_close"), avail.reset_index(names="avail"),
                      left_on="bar_close", right_on="avail", direction="backward")
    m.index = bars
    res = m[["composite", "alert", "confirm"]].rename(columns=lambda c: "oc_" + c)
    res["oc_alert"] = res["oc_alert"].fillna(False).astype(bool)
    res["oc_confirm"] = res["oc_confirm"].fillna(False).astype(bool)
    return res


# ---------------------------------------------------------------------------
# Comprobación previa: ¿la alerta anticipa caídas? (antes de integrarla)
# ---------------------------------------------------------------------------
def _fwd_stats(close: pd.Series, horizon: int) -> pd.DataFrame:
    fwd_ret = close.shift(-horizon) / close - 1
    fut_min = close[::-1].rolling(horizon, min_periods=horizon).min()[::-1].shift(-1)
    fwd_dd = fut_min / close - 1
    return pd.DataFrame({"ret": fwd_ret, "dd": fwd_dd})


def onchain_check(close_daily: pd.Series, sig: pd.DataFrame, lag_hours: int,
                  horizons=(7, 14, 30), n_perm: int = 1000, seed: int = 7) -> pd.DataFrame:
    """Compara rentabilidad y caída máxima futuras tras días con alerta vs sin ella.
    p-valor por desplazamiento circular de la serie de alertas (respeta autocorrelación)."""
    rng = np.random.default_rng(seed)
    # Vela diaria D cierra en D+1 00:00. El dato de D está en D+1 + lag:
    # con lag 0 actúas al cierre de la vela D; con lag > 0, al cierre de la vela D+1.
    shift_days = 0 if lag_hours == 0 else 1
    s = sig[["alert", "composite"]].copy()
    s.index = s.index + pd.Timedelta(days=shift_days)
    close = close_daily.copy()
    close.index = close.index.normalize()
    df = pd.concat([close.rename("close"), s], axis=1, join="inner").dropna(subset=["composite"])
    rows = []
    for h in horizons:
        st = _fwd_stats(df["close"], h)
        d = pd.concat([df["alert"], df["composite"], st], axis=1).dropna()
        a = d["alert"].astype(bool).values
        if a.sum() < 5:
            continue
        diff = d["dd"][a].mean() - d["dd"][~a].mean()
        perm = []
        n = len(d)
        for _ in range(n_perm):
            k = rng.integers(60, n - 60)
            ap = np.roll(a, k)
            perm.append(d["dd"].values[ap].mean() - d["dd"].values[~ap].mean())
        p = float((np.array(perm) <= diff).mean())   # una cola: alerta -> caída mayor
        rows.append({
            "horizonte_d": h,
            "dias_alerta": int(a.sum()),
            "ret_medio_alerta": d["ret"][a].mean(),
            "ret_medio_resto": d["ret"][~a].mean(),
            "dd_medio_alerta": d["dd"][a].mean(),
            "dd_medio_resto": d["dd"][~a].mean(),
            "P(dd<-15%)_alerta": (d["dd"][a] < -0.15).mean(),
            "P(dd<-15%)_resto": (d["dd"][~a] < -0.15).mean(),
            "spearman_comp_ret": d["composite"].corr(d["ret"], method="spearman"),
            "p_valor_dd": p,
        })
    return pd.DataFrame(rows)
