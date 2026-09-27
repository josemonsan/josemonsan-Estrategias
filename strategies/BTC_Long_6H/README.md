# BTC_Long_6H

Estrategia tendencial solo largos para BTC en velas de 6H. Esta es la versión archivada (congelada)
el 2026-09-27 de `pine/macd_d1_sma800_6h.pine`, con los parámetros recomendados en
`notes/2026-09-26-macd-sma800-6h-btc-optimization.md`.

| Archivo | Contenido |
|---|---|
| `BTC_Long_6H.pine` | Script Pine v5, con los defaults de esta configuración |
| `tv_trades_INDEX_BTCUSD_6H_2013-2026.csv` | Export "List of trades" de TradingView con esta configuración |

## Lógica

- **Régimen:** MACD diario (EMA12 − EMA26) de la vela D1 cerrada anterior > 0.
- **Entrada larga** (con régimen on y ADX(14) > 20):
  - cruce del cierre al alza sobre la SMA 800 (6H), equivalente a la SMA200 diaria, o
  - con el cierre por encima de la SMA, ruptura del máximo de las 20 velas anteriores.
- **Salida:**
  - stop inicial de 3 × ATR(14) D1;
  - Chandelier de 8 × ATR(14) D1 desde el máximo;
  - cierre a mercado cuando el régimen D1 se apaga.
- **Ejecución:** 100 % del equity por operación, comisión de 0,0035 % y 1 tick de deslizamiento. Sin
  lookahead: los datos D1 usan solo velas cerradas.

## Configuración (defaults del `.pine`)

| Input | Valor |
|---|---|
| Dirección | Solo largos |
| A · Filtro ADX | on (14, > 20) |
| S · Filtro pendiente SMA | off |
| D · Stop y trail con ATR D1 | on |
| R · Reentrada por ruptura | on (20 velas) |
| P · Cierre parcial | off |
| H · No salir por régimen en tendencia | off |
| Stop inicial / Chandelier | 3 × ATR / 8 × ATR |
| Salir si el régimen D1 se apaga | on |

## Resultado en TradingView (`INDEX:BTCUSD`, 6H, 2013-01-01 → 2026-09-27, capital inicial 10.000 USD)

| Métrica | Valor |
|---|---|
| Operaciones cerradas | 39 (más 1 abierta desde 2026-08-20, +18,1 %) |
| Ganadoras | 24 (61,5 %) |
| Profit factor | 3,53 |
| PnL neto cerrado | +127.075 % |
| PnL incluyendo la abierta | +150.123 % |
| Retorno medio por operación | +34,2 % |
| Max DD sobre operaciones cerradas | 54,6 % (dic 2013 → jul 2014, era Mt.Gox) |

Salidas: 29 por "Régimen off" (media +29,7 %) y 10 por Trail (media +47,1 %).

## Validación

- La réplica `macd_sma800_6h_backtest.py` reproduce 20 de las 21 entradas de TradingView desde 2020. La
  que falta es la posición abierta de 2026, y la diferencia se debe al feed de datos (Bitstamp frente a
  INDEX). Se reproduce con `python3 macd_sma800_6h_backtest.py eval`.
- Robustez (3 feeds, meseta de parámetros, walk-forward): ver
  `notes/2026-09-26-macd-sma800-6h-btc-optimization.md`.
- Limitaciones en mercado alcista (2024–25) y la variante opcional H: ver
  `notes/2026-09-27-macd-sma800-6h-bull-2024-2025.md`.

## Avisos

- Hay pocas operaciones (unas 20 desde 2020) y la estrategia se ha optimizado sobre un solo activo con una
  tendencia secular alcista. Es un resultado in-sample, no una validación para operar en real.
- Al estar 100 % invertida, los drawdowns intra-operación pueden superar el DD de operaciones cerradas.
