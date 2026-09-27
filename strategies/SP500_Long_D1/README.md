# SP500_Long_D1

Estrategia **BTC_Long_6H** (`strategies/BTC_Long_6H`) aplicada a cada valor del S&P 500 y optimizada
por valor con `sp500_optimize.py`. Hay un script Pine por valor: `<TICKER>_Long_D1.pine` (los tickers
con punto, como BRK.B, usan guion bajo: `BRK_B_Long_D1.pine`). En total son 458 valores; los que
empezaron a cotizar después de 2015 se excluyen porque no tienen periodo de entrenamiento.

## Adaptación a acciones

- **Timeframe diario (1D), no 6H.** Yahoo solo sirve datos intradía de los últimos ~730 días, que no
  alcanzan para calentar una SMA800 de 6H (~400 días) y hacer un backtest. Se usan longitudes
  equivalentes: SMA 200 D1 (= SMA800 6H en cripto), pendiente de 5 días y ruptura de 5 días (= 20 velas
  6H). La lógica Pine es idéntica; solo cambian los defaults.
- Son solo largos, igual que BTC_Long_6H. Comisión de 0,0035 %, 1 tick de deslizamiento y 100 % del
  equity por operación.

## Optimización

- **Rejilla por valor** (432 combinaciones):
  - SMA [150, 200, 250];
  - Stop [2.0, 3.0, 4.0] × ATR;
  - Chandelier [4.0, 6.0, 8.0] × ATR;
  - ADX off/15/20/25;
  - S on/off;
  - H on/off.
- **Selección:** Calmar (CAGR / maxDD) suavizado con los vecinos de la rejilla (meseta, no pico), con un
  mínimo de 5 operaciones.
- **Test honesto:** los parámetros se eligen solo con 2006–2018 y se miden en 2019–hoy (fuera de muestra).
- **Parámetros del `.pine`:** se eligen igual pero con todo 2006–hoy. Su rendimiento futuro esperado es el
  fuera de muestra de abajo, no el in-sample de la cabecera de cada script.

## Resultado fuera de muestra (2019 → hoy, mediana entre valores)

| Configuración | CAGR | maxDD | Calmar | Valores con CAGR > 0 | Valores que baten B&H |
|---|---|---|---|---|---|
| Optimizada por valor (elegida en 2006-18) | 2.0 % | 34.8 % | 0.06 | 66% | 10% |
| Universal BTC_Long_6H (SMA 200 · Stop 3 · Ch 8 · ADX>20) | 1.6 % | 33.8 % | 0.05 | 62% | 10% |
| Mejor config única del índice en 2006-18 (SMA 150 · Stop 3 · Ch 8 · ADX off · S off · H on) | 3.5 % | 40.8 % | 0.09 | 69% | 10% |
| Buy & hold | 12.5 % | — | — | 92% | — |

Optimizar por valor bate a la configuración universal en el 57% de
los valores. Todo el detalle está en `optimization_results.csv`: parámetros elegidos, in-sample 2006–hoy,
fuera de muestra y B&H.

## Parámetros elegidos (distribución entre valores)

| Parámetro | Reparto |
|---|---|
| smaLen | 150: 43% · 200: 22% · 250: 34% |
| sl | 2: 26% · 3: 29% · 4: 45% |
| ch | 4: 18% · 6: 22% · 8: 60% |
| adxThr | 0: 45% · 15: 16% · 20: 18% · 25: 21% |
| slope | off: 51% · on: 49% |
| hold | off: 26% · on: 74% |

## Top 25 fuera de muestra (con la columna `robust` del CSV)

`robust` = CAGR fuera de muestra > 0, ≥ el de la configuración universal y ≥ 10 operaciones en 2006–hoy.

| Ticker | Sector | CAGR OOS | maxDD OOS | CAGR B&H | Params del .pine (2006–hoy) |
|---|---|---|---|---|---|
| MU | Information Technology | 41.8 % | 49 % | 57.7 % | SMA 200 · Stop 2 · Ch 8 · ADX >25 · S on · H on |
| FIX | Industrials | 41.2 % | 40 % | 61.0 % | SMA 200 · Stop 3 · Ch 8 · ADX off · S on · H on |
| LRCX | Information Technology | 36.6 % | 42 % | 51.6 % | SMA 200 · Stop 4 · Ch 6 · ADX off · S off · H on |
| STX | Information Technology | 30.0 % | 45 % | 55.7 % | SMA 150 · Stop 2 · Ch 8 · ADX off · S off · H on |
| MPC | Energy | 27.4 % | 30 % | 31.6 % | SMA 150 · Stop 3 · Ch 6 · ADX >15 · S off · H on |
| GRMN | Consumer Discretionary | 24.3 % | 31 % | 24.9 % | SMA 250 · Stop 4 · Ch 8 · ADX off · S off · H on |
| KLAC | Information Technology | 23.8 % | 41 % | 49.8 % | SMA 150 · Stop 3 · Ch 8 · ADX off · S off · H on |
| GOOGL | Communication Services | 22.9 % | 23 % | 27.6 % | SMA 250 · Stop 4 · Ch 4 · ADX off · S off · H on |
| GOOG | Communication Services | 22.7 % | 19 % | 27.6 % | SMA 200 · Stop 4 · Ch 4 · ADX off · S off · H on |
| AVGO | Information Technology | 21.7 % | 39 % | 44.1 % | SMA 250 · Stop 2 · Ch 8 · ADX off · S on · H on |
| DECK | Consumer Discretionary | 20.7 % | 39 % | 18.2 % | SMA 250 · Stop 4 · Ch 8 · ADX off · S off · H on |
| TSLA | Consumer Discretionary | 20.2 % | 40 % | 45.3 % | SMA 250 · Stop 3 · Ch 4 · ADX >25 · S off · H off |
| FICO | Information Technology | 20.1 % | 39 % | 22.0 % | SMA 150 · Stop 4 · Ch 4 · ADX >20 · S on · H off |
| GNRC | Industrials | 19.4 % | 38 % | 19.9 % | SMA 150 · Stop 2 · Ch 8 · ADX off · S off · H on |
| AAPL | Information Technology | 18.9 % | 29 % | 33.1 % | SMA 150 · Stop 3 · Ch 4 · ADX off · S off · H off |
| WSM | Consumer Discretionary | 18.2 % | 36 % | 36.1 % | SMA 150 · Stop 3 · Ch 8 · ADX >25 · S off · H on |
| JCI | Industrials | 17.0 % | 30 % | 25.4 % | SMA 150 · Stop 3 · Ch 8 · ADX >25 · S off · H on |
| TGT | Consumer Staples | 16.9 % | 29 % | 15.0 % | SMA 250 · Stop 3 · Ch 8 · ADX >15 · S off · H on |
| APO | Financials | 16.3 % | 30 % | 26.6 % | SMA 250 · Stop 3 · Ch 6 · ADX off · S on · H on |
| SPG | Real Estate | 15.9 % | 19 % | 8.5 % | SMA 250 · Stop 3 · Ch 8 · ADX >15 · S off · H on |
| TT | Industrials | 15.9 % | 23 % | 28.9 % | SMA 150 · Stop 2 · Ch 6 · ADX >15 · S off · H on |
| IDXX | Health Care | 15.6 % | 30 % | 14.5 % | SMA 200 · Stop 3 · Ch 8 · ADX >20 · S off · H on |
| MPWR | Information Technology | 15.2 % | 48 % | 38.7 % | SMA 150 · Stop 2 · Ch 6 · ADX off · S off · H off |
| ERIE | Financials | 15.0 % | 22 % | 9.4 % | SMA 150 · Stop 3 · Ch 8 · ADX >15 · S on · H on |
| GLW | Information Technology | 15.0 % | 25 % | 27.1 % | SMA 150 · Stop 3 · Ch 4 · ADX off · S on · H on |

## Avisos

- **Sesgo de supervivencia:** se usa la lista actual del S&P 500. Los valores que salieron del índice (los
  perdedores) no están, así que todos los resultados, B&H incluido, salen inflados.
- **Sobreajuste:** hay 432 combinaciones por valor y
  ~35 operaciones por valor en 20 años. El dato que importa es la comparación fuera
  de muestra entre optimizar por valor y usar una configuración única.
- Estrategia 100 % invertida en un solo valor: no es una cartera. Los drawdowns individuales son altos.
- Los scripts no se han ejecutado uno a uno en TradingView. Los números vienen de la réplica Python
  (validada contra TradingView en BTC).
