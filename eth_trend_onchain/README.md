# ETH trend following + on-chain

Evalúa la estrategia ETH acordada: régimen SMA + ruptura Donchian, piramidación tipo Tortugas
(hasta 4 unidades, 1 % de riesgo por unidad, stop común a 2 ATR de la última entrada, techo x3)
y salida adaptativa con una alerta on-chain construida con datos públicos de BigQuery.

Compara siempre **A (sin on-chain)** contra **B (con on-chain)** y contra buy & hold.

## Instalación

```bash
pip install -r requirements.txt
gcloud auth application-default login     # solo para el paso on-chain
```

## Flujo

```bash
# 0. Comprobar que todo corre (datos SINTÉTICOS, los resultados no significan nada)
python main.py demo

# 1. Precios (okx = perpetuo desde ~2019-12; binance = spot desde 2017-08)
python main.py prices --exchange binance --timeframe 1d --since 2017-08-01 --out data/eth_1d.csv
#    También vale un CSV exportado de TradingView (columnas time, open, high, low, close)

# 2. On-chain (hace un dry-run y te dice los GB antes de gastar cuota; plan gratis 1 TB/mes)
python main.py onchain --project TU_PROYECTO_GCP --exchanges exchange_addresses.csv --out data/onchain_daily.csv

# 3. ¿La alerta anticipa caídas? Hazlo ANTES de integrarla
python main.py check --prices data/eth_1d.csv --onchain data/onchain_daily.csv

# 4. Backtest A/B + gráfico interactivo
python main.py backtest --prices data/eth_1d.csv --onchain data/onchain_daily.csv --html out/bt.html

# 5. Walk-forward (3 años train / 1 test) + Deflated Sharpe con 16 trials
python main.py walkforward --prices data/eth_1d.csv --onchain data/onchain_daily.csv
```

Funding real: `--funding funding.csv` (columnas `time, rate`). Sin él se usa 0,01 % cada 8 h.

## Módulos

| Archivo | Qué hace |
|---|---|
| `config.py` | Parámetros pre-registrados y rejilla. Congélalo antes de mirar OOS |
| `data.py` | Descarga ccxt, carga CSV (también TradingView), funding por barra |
| `onchain.py` | SQL BigQuery, z-scores causales, alerta/confirmación, alineado sin anticipación, test previo |
| `strategy.py` | Motor de backtest barra a barra |
| `metrics.py` | Métricas, DSR (Bailey–López de Prado), walk-forward |
| `report.py` | HTML Plotly: precio y operaciones, equity log, drawdown, compuesto on-chain |
| `../pine/eth_trend_onchain_v1.pine` | Traducción a Pine (TradingView) para verificación visual de entradas/salidas/piramidación; carga el preset 1d o 4h reoptimizado (sin on-chain) |

## Supuestos a conocer

- Señales al cierre, ejecución a la apertura siguiente. Stops y añadidos, dentro de la barra.
- Si en una barra se tocan stop y añadido, primero el stop; si tras añadir el mínimo toca el nuevo
  stop, se asume que saltó (conservador).
- El dato on-chain del día D se usa desde D+1 06:00 UTC (`lag_hours`).
- `vdd` es la versión de *coin days destroyed* para un modelo de cuentas: ETH enviado × días que la
  dirección llevaba sin enviar. Es una aproximación, no el CDD exacto de BTC.
- Solo transferencias de ETH de primer nivel (tabla `transactions`): no incluye internas ni WETH.
- La alerta on-chain solo actúa sobre largos. Los cortos siguen el Donchian.
- Las etiquetas de exchanges se añaden a posteriori: pueden introducir sesgo de anticipación.
