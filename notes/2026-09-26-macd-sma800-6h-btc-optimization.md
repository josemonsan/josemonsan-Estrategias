# MACD D1 + SMA800 6H — optimización para BTC

**Fecha:** 2026-09-26
**Estrategia:** `pine/macd_d1_sma800_6h.pine` (v2, recibida del usuario).
**Réplica/optimizador:** `macd_sma800_6h_backtest.py` (`fetch`, `eval`, `optimize`).
**Restricción del usuario:** A (filtro ADX) activado y P (cierre parcial) desactivado.

## Resultado

| Config (2020-01-03 → 2026-09-26) | Neto | Max DD | Ops | PF |
|---|---|---|---|---|
| Tu captura en TradingView (Stop 3, Chandelier 6, ADX on) | +838 % | 32,2 %* | 26 | 2,50 |
| Misma config en la réplica (Binance / Bitstamp / Coinbase) | +620 / +864 / +857 % | 48 % | 30–32 | 2,0–2,1 |
| **Recomendada** en la réplica (Binance / Bitstamp / Coinbase) | **+1722 / +1700 / +1699 %** | **31,3–31,6 %** | 20 | 3,4 |

\* TradingView y la réplica no miden el DD igual y la lista de operaciones no coincide exactamente
(26 frente a 30–32, porque el feed y el historial cargado son distintos). Compara siempre dentro de la
misma herramienta: réplica contra réplica, TV contra TV.

**Configuración recomendada** (ya puesta como defaults en el `.pine`):

| Input | Antes | Ahora |
|---|---|---|
| Dirección | Ambas | **Solo largos** |
| A · Filtro ADX | off | **on** (14, >20) |
| S · Filtro pendiente SMA | on | **off** |
| P · Cierre parcial | on | **off** |
| Stop inicial (× ATR) | 2 | **3** |
| Chandelier (× ATR) | 3 | **8** |
| SMA, D (ATR D1), R (ruptura 20), salida por régimen | — | sin cambios |

## Qué aporta cada cambio (ablación desde tu config, feed Bitstamp)

| Cambio | Neto | DD | Ops |
|---|---|---|---|
| Base (sl 3, ch 6, ADX on) | +864 % | 48,3 % | 30 |
| + Chandelier 8 | +871 % | 48,3 % | 30 |
| + S off | +1119 % | 53,1 % | 40 |
| + Solo largos | +841 % | 31,1 % | 19 |
| + Chandelier 8 + S off | +1252 % | 53,1 % | 40 |
| **Los tres juntos** | **+1700 %** | **31,4 %** | 20 |

- **Solo largos** es el cambio que más reduce el DD: los cortos perdieron en 2022 (−8 k y −3,6 k
  seguidos en feb–mar) y en el holdout de 2018–19 apenas sumaron. Es también el cambio más
  dependiente de la deriva alcista de BTC: la estrategia queda plana en mercados bajistas, no los explota.
- **S off**, por sí solo, sube el DD. Combinado con Solo largos deja entrar antes en las
  tendencias alcistas, y el filtro de régimen MACD D1 ya protege el lado largo.
- **Chandelier ≥ 7× ATR D1 no llega a saltar nunca en 2020–2026.** Las salidas son el stop inicial o
  el cambio de régimen (MACD D1 < 0). Con 8 el resultado es idéntico a 10, 12 o 15. En la práctica,
  la salida es "régimen off"; el Chandelier solo es un seguro para caídas extremas.

## Robustez

- **Rejilla:** 46.080 combinaciones (sl, ch, ADX umbral/periodo, SMA 600/800/1000, pendiente,
  ruptura, toggles S/R/D/régimen, dirección), evaluadas en 3 feeds de BTC.
- **Meseta, no pico:** la recomendada tiene vecinos casi igual de buenos:
  - sl 2,5–4: +1429 % a +1700 %
  - SMA 700–1000: +1455 % a +1700 %
  - ADX 20–25: +1528 % a +1700 %
  - ruptura 10–40: +1684 % a +1700 %

  El máximo bruto de la rejilla ("Ambas", SMA 1000, pendiente 40, ruptura 10: +2180 %) es un pico
  estrecho. Su peor vecino cae a Calmar 0,71, frente a 1,31 del peor vecino de la recomendada.
- **Walk-forward:** si se eligen parámetros solo con 2020-01 → 2023-06, las 200 mejores quedan en el
  percentil ~85 de 2023-07 → hoy. La moda de ese top-200 es casi exactamente la recomendada
  (sl 2–3, ch 8, ADX 20/14, SMA 800, S off, R on, D on, régimen on, solo largos).
- **Holdout 2018-06 → 2019-12** (fuera de la ventana de TV): +100 % con 1 operación (rally de 2019).
  Estuvo plana todo el bajista de 2018. Es positivo, pero con 1 operación no demuestra nada.

## Avisos

- **20 operaciones en 6,7 años.** Cada operación pesa mucho. Una o dos operaciones distintas en otro
  feed pueden mover el neto cientos de puntos.
- **Todo es in-sample sobre un único activo** con una tendencia secular fuerte. La mejora de "Solo
  largos" es, en buena parte, una apuesta por que esa tendencia continúe.
- La réplica no implementa el cierre parcial (P). Se optimizó con P off, como pediste.
- La réplica no coincide operación a operación con TradingView. Para calibrarla exactamente hacen falta
  el símbolo exacto de TV y la lista de operaciones exportada ("List of trades" → CSV).
