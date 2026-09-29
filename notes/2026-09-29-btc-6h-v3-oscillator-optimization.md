# MACD-SMA800 v3 — optimización de RSI (M/O) y Estocástico (K) en BTC 6H

**Fecha:** 2026-09-29
**Estrategia:** `pine/macd_sma800_6h_v3.pine` (MACD D1 régimen + SMA800 6H entrada + Chandelier ATR D1 salida).
**Alcance:** solo los módulos nuevos de la v3. Todo lo de v2 queda congelado en sus defaults (Solo largos, ADX>20, D, R=20, SL 3×ATR, Chandelier 8×ATR, salida por régimen).

**Conclusión:** ninguno de los tres módulos aporta una mejora que se distinga del ruido. M es redundante, O es perjudicial y K solo ayuda marginalmente con parámetros lentos (28/5/10), apoyándose en 2 operaciones de 2019. **Recomendación: mantener M, O y K apagados**, que es la v2. Si se quiere probar K, usar 28/5/10 (nuevos defaults en la Pine) y tratarlo como no validado.

---

## Método

- **Sin API de TradingView:** el Strategy Tester no se puede lanzar desde fuera. La estrategia se portó a Python (`btc/macd_sma800_v3.py`) replicando el emulador de TV: señales al cierre, órdenes de mercado a la apertura siguiente, stops activos desde la vela siguiente con llenado en gap, comisión 0,0035 %/lado, slippage 1 tick, qty = 100 % equity / cierre de señal, D1 con la vela diaria cerrada anterior. El cierre parcial (P) no está portado (apagado por defecto).
- **Datos:** Binance BTCUSDT spot 6H, 2017-08-17 → 2026-09-29 (`btc/fetch_btc.py`, `data/btcusdt_6h.csv`). Con SMA800 la primera operación es 2018-03. **Si el gráfico de TV usa otro feed (BITSTAMP, CFD del bróker, etc.), los números cambiarán algo.**
- **Rejilla** (`btc/optimize_v3.py`, 18.316 combinaciones deduplicadas):
  - `rsiLen` ∈ {7, 10, 14, 21, 28}
  - M `rsiMin` ∈ {off, 40…65 en pasos de 5}
  - O `rsiMax` ∈ {off, 65…90 en pasos de 5}
  - K ∈ {off} ∪ `stLen` {5, 9, 14, 21, 28} × `stSmooth` {1, 3, 5} × `stOS` {10, 15, 20, 25, 30}
- **Validación:** IS = 2018–2022 (selección), OOS = 2023–2026-09, y 4 bloques de ~2 años. Sharpe con retornos diarios de la equity (√365).

## Baseline (v3 con todo apagado == v2)

| | Neto % | CAGR % | Max DD % | Sharpe | Trades | Win % | PF |
|---|---|---|---|---|---|---|---|
| 2018–2026 | +3.779 | 49,4 | 35,6 | 1,30 | 22 | 59 | 3,52 |
| IS 2018–22 | +1.157 | 60,2 | 35,6 | 1,42 | 8 | 75 | 48 |
| OOS 2023–26 | +209 | 35,1 | 31,3 | 1,11 | 14 | 50 | 2,64 |

Por bloque, Sharpe: 2018-20 1,67 · 2021-22 1,05 · 2023-24 1,54 · **2025-26 0,35** (PF 0,69: las rupturas de 2025 son el punto débil actual).

**Comprobación recomendada:** cargar la v3 con defaults en TV sobre BINANCE:BTCUSDT 6H y comparar con la tabla (22 operaciones, ~+3.779 %). Si coincide, el port es fiable para tu feed.

## Resultados por módulo

### M — RSI momentum (mínimo)
- Sharpe total 1,26–1,31 en las 30 combinaciones (baseline 1,30). 21–22 operaciones en casi todas.
- **Motivo:** un cierre por encima de la SMA800 o una ruptura del máximo de 20 velas ya implica RSI > 50 casi siempre; el filtro no filtra nada.
- Solo con `rsiMin` = 65 hace algo: con RSI 14 quita 2 operaciones (2018-03 +4,6 %, 2025-09 −6,2 %) → Sharpe 1,31; con RSI 28 retrasa/sustituye 10 operaciones, mejora 2025-26 (0,35 → 0,78) pero empeora 2023-24 (1,54 → 1,35). No es robusto.

### O — RSI techo (no perseguir)
- **Peor que el baseline en las 30 combinaciones, tanto IS como OOS.** Con el default (14, 75): Sharpe 1,17, neto +2.238 %, DD 37,9 %.
- **Motivo:** los mejores arranques de tendencia llegan con RSI alto. O no los anula, pero los retrasa a peor precio (p. ej. 2020-10: +335 % → +328 %; 2019-04: +103 % → +80 %) o los cambia por entradas peores.
- Con umbrales estrictos (65/70) y RSI corto se queda con 4–16 operaciones.

### K — Reentrada por pullback estocástico
- **Default 14/3/20: empeora** (Sharpe 1,23, DD 42,5 %). Añade 4 pullbacks en 2018-19, 3 de ellos perdedores.
- Meseta suave: mejor con `stLen` largo, suavizado 5 y sobreventa baja. Media por suavizado: Sharpe 1,23 (1) / 1,25 (3) / 1,28 (5).
- **Mejor: 28/5/10** → Sharpe 1,33, neto +4.139 %, DD 31,3 %, MAR 1,62 (baseline 1,39). Solo añade 2 operaciones (2019-07 +9,4 %, 2019-08 −0,1 %).
- **OOS 2023–26: ninguna operación pullback con estos parámetros** → OOS idéntico al baseline. La mejora es 100 % in-sample.

### Combinaciones
- **0 de 18.315** combinaciones superan al baseline en Sharpe **a la vez** en IS y OOS.
- La mejor global (M65/RSI14 + K 28/5/10) da Sharpe 1,34 frente a 1,30: **+0,03**, cuando la dispersión de la rejilla es σ = 0,19.
- La correlación de rangos IS→OOS (0,72) es alta solo porque la mayoría de combinaciones dan casi las mismas operaciones que el baseline; no indica poder predictivo.
- DSR del mejor (N = 18.316) = 0,95. Ese valor lo explica la base v2 (Sharpe 1,30), no los módulos v3. Además la v2 ya se optimizó sobre estos mismos datos, así que su propio DSR está sobreestimado.

## Limitaciones
- **22 operaciones en 8,5 años:** cada módulo cambia 1–10 operaciones. Ninguna mejora de ese tamaño es estadísticamente distinguible.
- **Feed Binance spot:** el gráfico de TV del usuario puede ser otro (CFD con swap, otro exchange).
- **Supuestos de costes del script:** comisión 0,0035 % y swap 0. En spot Binance la comisión real es ~0,1 %/lado; con 22 operaciones el impacto es pequeño.
- **Max DD sobre equity al cierre** (como la tabla del script). El DD intrabar del resumen de TV será algo mayor.

## Pasos siguientes (si se quiere seguir)
- El problema real está en **2025-26 (Sharpe 0,35)**: varias rupturas R tardías acabaron en stop (2025-01, 2025-09, 2025-10). Merece más la pena estudiar R (p. ej. exigir distancia mínima a la SMA800 o limitar rupturas por régimen) que M/O/K, contando esas pruebas en el DSR.

## Reproducir
```
pip install numpy pandas
python3 btc/fetch_btc.py            # refresca data/btcusdt_6h.csv
python3 btc/macd_sma800_v3.py       # baseline + lista de operaciones
python3 btc/optimize_v3.py          # rejilla (~8 min, 4 núcleos) → btc/results/grid_v3.csv.gz
python3 btc/analyze_v3.py           # tablas de esta nota
```
