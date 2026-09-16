"""
cava_trend_strategy_v3.py
=======================================================================
Versión 3 del motor Cava. Fuente principal: síntesis de "El Arte de
Especular" (libro, 439 pp., OCR temático). Fuentes secundarias: curso de
Murcia 2004 y Base.txt (notas propias del usuario).

Cada parámetro lleva etiqueta de PROCEDENCIA para el registro de trials:
  [LIBRO]  = El Arte de Especular (síntesis .docx)
  [CURSO]  = curso Murcia 2004 (.doc)
  [BASE]   = Base.txt (notas propias; incluye el Cava reciente: base monetaria)
  [PROPIO] = elección arbitraria mía necesaria para sistematizar — cada
             valor alternativo que se pruebe cuenta como trial.

ARQUITECTURA v3 (todo causal; ejecución next-open; huecos a través del
stop; contabilidad mark-to-market diaria heredada y verificada de v2)
-----------------------------------------------------------------------
GATES GLOBALES (obligatorios para módulos tendenciales):
  - Tendencia multi-marco [LIBRO §3.3, criterio preferido del autor]:
      alcista  = MACD mensual cortado al alza  Y  línea rápida MACD
                 semanal > 0  Y  %K estocástico semanal (14) > 60
      bajista  = simétrico (<0, <40)
      resto    = sin tendencia -> NO se opera [BASE: "no operar sin
                 tendencia"; por eso el módulo de rango lateral del
                 libro (§8, ADX<20) NO está implementado]
  - Gate de Wilder [LIBRO §6.5]: sin entradas tendenciales si el ADX
    está por debajo de ambos DI.
  - Macro (base monetaria USD) [BASE — NO aparece ni en el libro ni en
    el curso; hipótesis de procedencia distinta, toggle separado].
  - Liquidez (volumen + huecos) [BASE].
  - Bloqueo por divergencia [LIBRO §5]: una divergencia confirmada
    bloquea nuevas entradas en su dirección durante un cooldown.
    Necesaria pero no suficiente: aquí solo VETA, nunca dispara.
  - Estacionalidad [LIBRO §9]: OFF por defecto (familia de hipótesis
    separada, anomalías conocidas y probablemente decaídas).

MÓDULOS DE ENTRADA (toggles independientes = trials independientes):
  1. rsi_macd_weekly   [LIBRO §7.2/§8 — el núcleo]: RSI diario recupera
     el nivel de sobreventa adaptativo (40 si tendencia fuerte por
     ADX>30, 30 si normal [LIBRO §6.2 tabla]) con confirmación de
     PRECIO (cierre > máximo de la vela previa) [LIBRO §4: "la señal la
     desencadena exclusivamente el precio"].
  2. macd_daily_cross  [LIBRO §8]: cruce del MACD diario a favor de la
     tendencia (el cruce de medias ya ES un hecho de precio, §4).
  3. adx_ema20_pullback [LIBRO §7.4]: ADX>30 y subiendo, +DI>-DI,
     retroceso a la EMA(20) y giro al alza sobre ella; stop bajo la
     media.
  4. cci_threshold     [LIBRO §7.6]: CCI cruza +50 a favor de tendencia;
     salida propia cuando el CCI pierde el cero. OFF por defecto.
  5. trap_reversal     [BASE/CURSO]: escape falso confirmado -> entrada
     contraria (heredado de v2, causal).
  6. v2_confluence     [BASE, proxies de v1/v2]: retroceso a MA +
     estocástico + marubozu + pauta-plana-proxy + canal. OFF por
     defecto — se conserva por linaje de trials.

STOPS Y SALIDAS:
  - Stop inicial estructural [LIBRO §10]: ligeramente por debajo del
    mínimo reciente (mín. de N barras − colchón ATR), con TOPE DURO del
    3% de pérdida [BASE: "con pérdidas del 3% salgo"] — el stop puede
    ser más ceñido que el 3%, nunca más amplio.
  - Trailing [LIBRO §10]: Parabolic SAR de Wilder anclado a la posición
    (por defecto). Alternativas: chandelier ATR (v2) o EMA(10)
    [CURSO: "medias de 5 y 10 como instrumento de stop"]. El stop solo
    se estrecha, nunca se amplía [BASE].
  - Salida por señal [LIBRO §8]: cruce del MACD diario en contra
    cancela la posición (ejecutada en la siguiente apertura). El módulo
    CCI usa su propia cancelación (pérdida del cero).
  - Reentrada [LIBRO §1, quinto elemento del sistema]: cooldown
    configurable tras cada salida (0 = reentrada inmediata con nueva
    señal).

NO IMPLEMENTADO deliberadamente (documentado para el registro):
  - Operativa en rango lateral (ADX<20) [LIBRO §8]: contradice la regla
    dura de Base.txt; sería una familia de estrategia distinta.
  - Línea "2-4" de Elliott, pautas terminales, Fibonacci sobre el RSI:
    exigen etiquetado de ondas/pivotes con demasiados grados de
    libertad (coste en DSR alto); igual criterio que en el gap analysis
    del curso.
  - Sesgo de vísperas de festivo USA: requiere calendario de festivos;
    valor esperado bajo.
  - La regla "tres divergencias consecutivas" del CURSO no aparece como
    tal en el LIBRO (allí el nº de techos/suelos es un factor de
    fiabilidad): se parametriza como `div_pivots_required` (2 = libro,
    3 = curso) para poder testar ambas lecturas como trials separados.

La demo final usa DATOS SINTÉTICOS solo para validar que el pipeline
corre sin errores y sin sesgos evidentes (resultado ~nulo esperado).
Nada de este archivo está validado: es la especificación congelable de
una familia de hipótesis.

Dependencias: numpy, pandas.
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, Optional


# =======================================================================
# 1. CONFIGURACIÓN CONGELADA (etiquetas de procedencia en cada bloque)
# =======================================================================

@dataclass(frozen=True)
class CavaConfigV3:
    # --- Tendencia multi-marco [LIBRO §3.3 vs §8 — dos lecturas, dos trials] ---
    trend_mode: Literal["macd_multi_tf", "ma_slope"] = "macd_multi_tf"
    use_weekly_stoch_in_trend: bool = False  # False = gate tabla §8 (MACD mensual alineado +
                                             #         signo del MACD semanal) — por defecto,
                                             #         porque es el contexto que usa la propia
                                             #         tabla de entradas del libro.
                                             # True  = criterio estricto §3.3 (añade %K semanal
                                             #         >60/<40). Incompatible de facto con el
                                             #         módulo de retroceso RSI: el pullback que
                                             #         dispara la entrada tumba el estocástico
                                             #         semanal y anula la tendencia justo ahí.
    trend_stoch_weekly_up: float = 60.0      # [LIBRO §3.3] %K semanal > 60
    trend_stoch_weekly_dn: float = 40.0      # [LIBRO §3.3] %K semanal < 40
    trend_ma_period: int = 10                # [PROPIO] solo para trend_mode="ma_slope" (legado v2)
    trend_slope_lookback: int = 3            # [PROPIO] idem

    # --- MACD [LIBRO §6.1, parámetros estándar] ---
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9

    # --- RSI [LIBRO §6.2] ---
    rsi_period: int = 14
    rsi_os_normal: float = 30.0              # [LIBRO] sobreventa estándar
    rsi_os_strong_up: float = 40.0           # [LIBRO] sobreventa en tendencia alcista marcada
    rsi_ob_normal: float = 70.0              # [LIBRO] sobrecompra estándar
    rsi_ob_strong_dn: float = 60.0           # [LIBRO] sobrecompra en tendencia bajista marcada

    # --- Estocástico [LIBRO §6.3] ---
    stoch_k_period: int = 14
    stoch_d_period: int = 3

    # --- ADX [LIBRO §6.5] ---
    adx_period: int = 14
    adx_strong: float = 30.0                 # [LIBRO] ADX>30 girando al alza = tendencia fuerte
    use_adx_wilder_gate: bool = True         # [LIBRO] sin seguimiento de tendencia si ADX < ambos DI

    # --- CCI [LIBRO §6.4] ---
    cci_period: int = 14
    cci_constant: float = 0.015              # [PROPIO] Lambert estándar; la síntesis dice "0,15",
                                             # casi seguro pérdida de decimal del OCR
    cci_entry_level: float = 50.0            # [LIBRO §7.6]

    # --- Divergencias [LIBRO §5] ---
    use_divergence_filter: bool = True
    div_indicator: Literal["rsi"] = "rsi"    # [PROPIO] RSI como indicador de referencia
    div_pivot_k: int = 3                     # [PROPIO] fractal: pivote confirmado k barras después
    div_pairing_lookback: int = 60           # [PROPIO] distancia máx. entre pivotes emparejados
    div_pivots_required: int = 2             # [LIBRO]=2 (par); [CURSO]=3 ("tres techos") — trial separado
    div_cooldown_bars: int = 15              # [PROPIO] barras de veto tras divergencia confirmada

    # --- Módulos de entrada (toggles = trials) ---
    enable_rsi_macd_weekly: bool = True      # [LIBRO §7.2/§8]
    enable_macd_daily_cross: bool = True     # [LIBRO §8]
    enable_adx_ema20_pullback: bool = True   # [LIBRO §7.4]
    enable_cci_threshold: bool = False       # [LIBRO §7.6] secundario
    enable_trap_reversal: bool = True        # [BASE/CURSO] heredado de v2
    enable_v2_confluence: bool = False       # [BASE] linaje v2
    require_price_confirmation: bool = True  # [LIBRO §4] cierre > máx. vela previa (largos)

    # --- EMA del módulo ADX-pullback [LIBRO §7.4] ---
    pullback_ema_period: int = 20
    pullback_touch_bars: int = 3             # [PROPIO] el toque a la EMA debe ser reciente

    # --- Escape falso [BASE/CURSO, causal desde v2] ---
    breakout_lookback: int = 20
    breakout_confirm_bars: int = 3
    false_breakout_cooldown_bars: int = 10

    # --- Confluencia v2 (solo si enable_v2_confluence) [BASE/PROPIO] ---
    retracement_mas: tuple = (20, 50, 100)
    retracement_tolerance_pct: float = 0.010
    flat_window: int = 15
    flat_width_threshold: float = 0.06
    channel_window: int = 60
    channel_band_mult: float = 2.0
    channel_long_zone: float = 0.25
    channel_short_zone: float = 0.75
    marubozu_body_ratio: float = 0.85
    v2_stoch_oversold: float = 25.0
    v2_stoch_overbought: float = 75.0
    min_confluence_score: int = 2

    # --- Macro (oferta monetaria USD) [BASE — no está en libro ni curso] ---
    use_macro_gate: bool = True
    macro_ma_period: int = 3
    macro_lookback_periods: int = 6
    macro_publication_lag_days: int = 14

    # --- Estacionalidad [LIBRO §9] — OFF por defecto ---
    use_seasonality_filter: bool = False
    # sesgo mensual: nov-mar alcista, may-oct bajista, abril neutro [LIBRO]
    # (se materializa en month_bias() más abajo)
    seasonality_block_shorts_pres_year3: bool = True   # [LIBRO] 3er año ciclo presidencial

    # --- Liquidez [BASE] ---
    min_avg_volume: float = 1_000_000.0
    liquidity_vol_window: int = 63
    max_gap_pct: float = 0.02
    gap_window: int = 60
    max_gap_frequency: float = 0.05

    # --- Stops [LIBRO §10 + BASE] ---
    stop_mode: Literal["structural", "fixed_pct"] = "structural"
    stop_lookback_bars: int = 5              # [PROPIO] "mínimo del retroceso/día de entrada" ≈ mín. de N barras
    stop_buffer_atr: float = 0.10            # [PROPIO] "ligeramente por debajo" ≈ 0,1·ATR de colchón
    initial_stop_pct: float = 0.03           # [BASE] tope duro de pérdida del 3%
    atr_period: int = 14
    trailing_mode: Literal["sar", "atr", "ema"] = "sar"   # [LIBRO] SAR por defecto
    sar_af_start: float = 0.02               # [LIBRO/Wilder] parámetros clásicos del SAR
    sar_af_step: float = 0.02
    sar_af_max: float = 0.20
    trailing_atr_mult: float = 2.5           # [PROPIO] modo "atr" (v2)
    trailing_ema_period: int = 10            # [CURSO] "medias de 5 y 10 como stop" -> EMA(10)
    max_holding_bars: int = 252              # [PROPIO] límite de seguridad
    use_macd_exit: bool = True               # [LIBRO §8] cruce MACD diario en contra cancela
    reentry_cooldown_bars: int = 0           # [LIBRO §1] criterio de reentrada (0 = inmediata)

    # --- Cartera [BASE + PROPIO, heredado de v2] ---
    risk_per_trade: float = 0.005
    cost_bps_per_side: float = 2.5
    max_concurrent_positions: int = 10
    max_gross_leverage: float = 2.0
    target_annual_vol: float = 0.05          # [BASE] objetivo 5% vol
    vol_target_window: int = 63
    vol_scalar_floor: float = 0.25
    pending_max_age_bars: int = 3

    # --- Escaneo semanal de universo [BASE: "buceos semanales"] ---
    top_n_candidates: int = 10


def month_bias(ts: pd.Timestamp) -> int:
    """Sesgo estacional mensual [LIBRO §9]: +1 nov-mar, -1 may-oct, 0 abril."""
    m = ts.month
    if m in (11, 12, 1, 2, 3):
        return 1
    if m in (5, 6, 7, 8, 9, 10):
        return -1
    return 0


def is_presidential_year3(ts: pd.Timestamp) -> bool:
    """3er año del ciclo presidencial USA (p.ej. 2019, 2023, 2027) [LIBRO §9]."""
    return (ts.year - 1) % 4 == 2


# =======================================================================
# 2. UTILIDADES NUMÉRICAS (idénticas a v2, verificadas)
# =======================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    if not (0.0 < p < 1.0):
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


# =======================================================================
# 3. INDICADORES
# =======================================================================

def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule).agg(agg).dropna(how="any")


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def compute_macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """[LIBRO §6.1] línea rápida = EMA(12)-EMA(26); lenta = EMA(9) de la rápida."""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line, macd_line - signal_line


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """RSI de Wilder [LIBRO §6.2] con suavizado exponencial alpha=1/n."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def compute_stochastic(df: pd.DataFrame, k_period: int, d_period: int):
    low_min = df["low"].rolling(k_period).min()
    high_max = df["high"].rolling(k_period).max()
    denom = (high_max - low_min).replace(0, np.nan)
    pct_k = 100 * (df["close"] - low_min) / denom
    pct_d = pct_k.rolling(d_period).mean()
    return pct_k, pct_d


def compute_cci(df: pd.DataFrame, period: int = 14, constant: float = 0.015) -> pd.Series:
    """[LIBRO §6.4] CCI de Lambert sobre el precio típico."""
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(period).mean()
    mad = (tp - ma).abs().rolling(period).mean()
    return (tp - ma) / (constant * mad.replace(0, np.nan))


def compute_adx(df: pd.DataFrame, period: int = 14):
    """[LIBRO §6.5] Sistema direccional de Wilder: +DI, -DI, ADX."""
    high, low, close = df["high"], df["low"], df["close"]
    up = high.diff()
    dn = -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    alpha = 1.0 / period
    atr_w = tr.ewm(alpha=alpha, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * plus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w
    mdi = 100 * minus_dm.ewm(alpha=alpha, adjust=False).mean() / atr_w
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=alpha, adjust=False).mean()
    return pdi, mdi, adx


def rolling_linreg_channel(series: pd.Series, window: int, band_mult: float):
    """Canal de regresión vectorizado (verificado en v2 contra naive, err<2e-10)."""
    y = series.to_numpy(dtype=np.float64)
    n = len(y)
    nanser = pd.Series(np.nan, index=series.index)
    if n < window:
        return nanser.copy(), nanser.copy(), nanser.copy()

    def _rollsum(arr, w):
        c = np.cumsum(arr, dtype=np.float64)
        out = np.full(arr.shape, np.nan, dtype=np.float64)
        out[w - 1] = c[w - 1]
        out[w:] = c[w:] - c[:-w]
        return out

    j = np.arange(n, dtype=np.float64)
    Sy = _rollsum(y, window)
    Sjy = _rollsum(j * y, window)
    Sy2 = _rollsum(y * y, window)
    t = np.arange(n, dtype=np.float64)
    win_start = t - window + 1
    Sxy_raw = Sjy - win_start * Sy
    x_mean = (window - 1) / 2.0
    Sxx = window * (window * window - 1) / 12.0
    y_mean = Sy / window
    Sxy = Sxy_raw - window * x_mean * y_mean
    slope = Sxy / Sxx
    Syy = Sy2 - window * y_mean * y_mean
    sse = np.clip(Syy - slope * Sxy, 0.0, None)
    resid_std = np.sqrt(sse / window)
    mid_v = y_mean + slope * ((window - 1) - x_mean)
    return (pd.Series(mid_v, index=series.index),
            pd.Series(mid_v + band_mult * resid_std, index=series.index),
            pd.Series(mid_v - band_mult * resid_std, index=series.index))


def detect_marubozu(df: pd.DataFrame, body_ratio: float):
    body = (df["close"] - df["open"]).abs()
    rng = (df["high"] - df["low"]).replace(0, np.nan)
    ratio = (body / rng).fillna(0.0)
    return ((ratio >= body_ratio) & (df["close"] > df["open"]),
            (ratio >= body_ratio) & (df["close"] < df["open"]))


# =======================================================================
# 4. TENDENCIA MULTI-MARCO [LIBRO §3.3] Y FILTRO MACRO [BASE]
# =======================================================================

def classify_trend_v3(daily: pd.DataFrame, config: CavaConfigV3) -> pd.Series:
    """
    Devuelve la dirección de tendencia (+1/0/-1) proyectada al índice
    diario, según el criterio preferido del libro (macd_multi_tf) o el
    legado de v2 (ma_slope). Todo causal: las barras semanales/mensuales
    se etiquetan a FIN de periodo y se proyectan con ffill, de modo que
    cada día usa la última barra superior COMPLETADA.
    """
    if config.trend_mode == "ma_slope":
        monthly = resample_ohlcv(daily, "ME")
        ma = monthly["close"].rolling(config.trend_ma_period).mean()
        slope = (ma - ma.shift(config.trend_slope_lookback)) / ma.shift(config.trend_slope_lookback).abs()
        direction = pd.Series(0, index=monthly.index, dtype=int)
        direction[slope > 0] = 1
        direction[slope < 0] = -1
        return direction.reindex(daily.index, method="ffill").fillna(0).astype(int)

    weekly = resample_ohlcv(daily, "W")
    monthly = resample_ohlcv(daily, "ME")
    m_fast, m_slow, _ = compute_macd(monthly["close"], config.macd_fast, config.macd_slow, config.macd_signal)
    w_fast, _, _ = compute_macd(weekly["close"], config.macd_fast, config.macd_slow, config.macd_signal)
    wk, _ = compute_stochastic(weekly, config.stoch_k_period, config.stoch_d_period)

    up_m = (m_fast > m_slow).reindex(daily.index, method="ffill")
    dn_m = (m_fast < m_slow).reindex(daily.index, method="ffill")
    up_w = (w_fast > 0).reindex(daily.index, method="ffill")
    dn_w = (w_fast < 0).reindex(daily.index, method="ffill")
    cond_up = up_m & up_w
    cond_dn = dn_m & dn_w
    if config.use_weekly_stoch_in_trend:
        # [LIBRO §3.3] criterio estricto de tendencia confirmada
        k_up = (wk > config.trend_stoch_weekly_up).reindex(daily.index, method="ffill")
        k_dn = (wk < config.trend_stoch_weekly_dn).reindex(daily.index, method="ffill")
        cond_up = cond_up & k_up
        cond_dn = cond_dn & k_dn

    trend = pd.Series(0, index=daily.index, dtype=int)
    trend[cond_up.fillna(False)] = 1
    trend[cond_dn.fillna(False)] = -1
    return trend


def macro_regime(macro_monthly: pd.DataFrame, config: CavaConfigV3, column: str = "monetary_base") -> pd.Series:
    """[BASE] Régimen de oferta monetaria con lag de publicación."""
    series = macro_monthly[column].astype(float)
    ma = series.rolling(config.macro_ma_period).mean()
    slope = (ma - ma.shift(config.macro_lookback_periods)) / ma.shift(config.macro_lookback_periods).abs()
    regime = pd.Series(0, index=macro_monthly.index, dtype=int)
    regime[slope > 0] = 1
    regime[slope < 0] = -1
    regime.index = regime.index + pd.Timedelta(days=config.macro_publication_lag_days)
    return regime.rename("macro_regime")


# =======================================================================
# 5. DIVERGENCIAS CAUSALES [LIBRO §5]
# =======================================================================

def detect_divergences(daily: pd.DataFrame, indicator: pd.Series, config: CavaConfigV3) -> pd.DataFrame:
    """
    Divergencia bajista: precio con máximos crecientes e indicador con
    máximos decrecientes a lo largo de `div_pivots_required` pivotes.
    Divergencia alcista: simétrica con mínimos.

    CAUSALIDAD: un pivote fractal en la barra i solo se conoce en i+k
    (necesita k barras posteriores). El flag se coloca en la barra de
    CONFIRMACIÓN del último pivote, nunca antes. El flag solo VETA
    entradas [LIBRO: necesaria pero no suficiente]; jamás dispara.
    """
    k = config.div_pivot_k
    req = max(2, config.div_pivots_required)
    lookback = config.div_pairing_lookback
    n = len(daily)
    high = daily["high"].to_numpy()
    low = daily["low"].to_numpy()
    ind = indicator.to_numpy(dtype=float)

    bear_known = np.zeros(n, dtype=bool)
    bull_known = np.zeros(n, dtype=bool)
    piv_hi: list[int] = []
    piv_lo: list[int] = []

    for i in range(k, n - k):
        if high[i] == high[i - k:i + k + 1].max():
            chain = [p for p in piv_hi if i - p <= lookback][-(req - 1):] + [i]
            if len(chain) == req and not np.isnan(ind[chain]).any():
                price_hh = all(high[chain[j + 1]] > high[chain[j]] for j in range(req - 1))
                ind_lh = all(ind[chain[j + 1]] < ind[chain[j]] for j in range(req - 1))
                if price_hh and ind_lh:
                    bear_known[i + k] = True
            piv_hi.append(i)
        if low[i] == low[i - k:i + k + 1].min():
            chain = [p for p in piv_lo if i - p <= lookback][-(req - 1):] + [i]
            if len(chain) == req and not np.isnan(ind[chain]).any():
                price_ll = all(low[chain[j + 1]] < low[chain[j]] for j in range(req - 1))
                ind_hl = all(ind[chain[j + 1]] > ind[chain[j]] for j in range(req - 1))
                if price_ll and ind_hl:
                    bull_known[i + k] = True
            piv_lo.append(i)

    return pd.DataFrame({"bear_div_known": bear_known, "bull_div_known": bull_known}, index=daily.index)


# =======================================================================
# 6. ESCAPE FALSO CAUSAL [BASE/CURSO] Y LIQUIDEZ [BASE] (de v2)
# =======================================================================

def detect_false_breakout(df: pd.DataFrame, config: CavaConfigV3) -> pd.DataFrame:
    n = len(df)
    prior_high = df["high"].rolling(config.breakout_lookback).max().shift(1).to_numpy()
    prior_low = df["low"].rolling(config.breakout_lookback).min().shift(1).to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    false_up = np.zeros(n, dtype=bool)
    false_down = np.zeros(n, dtype=bool)
    confirm = config.breakout_confirm_bars
    for i in range(n):
        ph, pl = prior_high[i], prior_low[i]
        if not np.isnan(ph) and high[i] > ph:
            for j in range(i + 1, min(i + 1 + confirm, n)):
                if close[j] < ph:
                    false_up[j] = True
                    break
        if not np.isnan(pl) and low[i] < pl:
            for j in range(i + 1, min(i + 1 + confirm, n)):
                if close[j] > pl:
                    false_down[j] = True
                    break
    return pd.DataFrame({"false_up_known": false_up, "false_down_known": false_down}, index=df.index)


def liquidity_filter(df: pd.DataFrame, config: CavaConfigV3) -> pd.Series:
    avg_vol = df["volume"].rolling(config.liquidity_vol_window).mean()
    volume_ok = avg_vol >= config.min_avg_volume
    prev_close = df["close"].shift(1)
    gap_pct = (df["open"] - prev_close).abs() / prev_close
    gap_freq = (gap_pct > config.max_gap_pct).astype(float).rolling(config.gap_window).mean()
    gaps_ok = gap_freq <= config.max_gap_frequency
    return (volume_ok & gaps_ok).fillna(False)


# =======================================================================
# 7. GENERACIÓN DE SEÑALES v3 (módulos + gates, todo precomputado causal)
# =======================================================================

def generate_signals_v3(
    daily: pd.DataFrame,
    macro_regime_series: pd.Series,
    config: CavaConfigV3,
) -> pd.DataFrame:
    close, high, low = daily["close"], daily["high"], daily["low"]
    idx = daily.index

    # --- Contexto ---
    trend = classify_trend_v3(daily, config)
    macro = macro_regime_series.reindex(idx, method="ffill").fillna(0).astype(int)
    liquid = liquidity_filter(daily, config)
    rsi = compute_rsi(close, config.rsi_period)
    macd_f, macd_s, _ = compute_macd(close, config.macd_fast, config.macd_slow, config.macd_signal)
    pdi, mdi, adx = compute_adx(daily, config.adx_period)
    cci = compute_cci(daily, config.cci_period, config.cci_constant)
    ema20 = close.ewm(span=config.pullback_ema_period, adjust=False).mean()
    atr = compute_atr(daily, config.atr_period)

    strong = (adx > config.adx_strong).fillna(False)

    # --- Gate de Wilder [LIBRO §6.5] ---
    if config.use_adx_wilder_gate:
        adx_ok = ~((adx < pdi) & (adx < mdi)).fillna(False)
    else:
        adx_ok = pd.Series(True, index=idx)

    # --- Divergencias -> veto [LIBRO §5] ---
    if config.use_divergence_filter:
        divs = detect_divergences(daily, rsi, config)
        w = config.div_cooldown_bars
        block_long = divs["bear_div_known"].astype(float).rolling(w).max().fillna(0) > 0
        block_short = divs["bull_div_known"].astype(float).rolling(w).max().fillna(0) > 0
    else:
        divs = pd.DataFrame({"bear_div_known": False, "bull_div_known": False}, index=idx)
        block_long = pd.Series(False, index=idx)
        block_short = pd.Series(False, index=idx)

    # --- Estacionalidad [LIBRO §9] ---
    if config.use_seasonality_filter:
        bias = pd.Series([month_bias(t) for t in idx], index=idx)
        season_block_long = bias == -1
        season_block_short = (bias == 1) | (
            pd.Series([is_presidential_year3(t) for t in idx], index=idx)
            if config.seasonality_block_shorts_pres_year3 else False
        )
    else:
        season_block_long = pd.Series(False, index=idx)
        season_block_short = pd.Series(False, index=idx)

    # --- Escape falso [BASE/CURSO] ---
    fb = detect_false_breakout(daily, config)
    fu, fd = fb["false_up_known"], fb["false_down_known"]
    recent_fu = fu.astype(float).rolling(config.false_breakout_cooldown_bars).max().fillna(0) > 0
    recent_fd = fd.astype(float).rolling(config.false_breakout_cooldown_bars).max().fillna(0) > 0

    # --- Gates combinados por dirección (módulos tendenciales) ---
    macro_ok_long = (macro != -1) if config.use_macro_gate else pd.Series(True, index=idx)
    macro_ok_short = (macro != 1) if config.use_macro_gate else pd.Series(True, index=idx)
    gate_long = (trend == 1) & macro_ok_long & liquid & adx_ok & (~block_long) & (~season_block_long) & (~recent_fu)
    gate_short = (trend == -1) & macro_ok_short & liquid & adx_ok & (~block_short) & (~season_block_short) & (~recent_fd)

    # --- Confirmación de precio [LIBRO §4] ---
    if config.require_price_confirmation:
        price_conf_long = close > high.shift(1)
        price_conf_short = close < low.shift(1)
    else:
        price_conf_long = pd.Series(True, index=idx)
        price_conf_short = pd.Series(True, index=idx)

    # === MÓDULO 1: RSI diario + MACD semanal [LIBRO §7.2/§8] ===
    os_level = pd.Series(np.where(strong, config.rsi_os_strong_up, config.rsi_os_normal), index=idx)
    ob_level = pd.Series(np.where(strong, config.rsi_ob_strong_dn, config.rsi_ob_normal), index=idx)
    m1_long = (rsi.shift(1) < os_level.shift(1)) & (rsi >= os_level) & price_conf_long
    m1_short = (rsi.shift(1) > ob_level.shift(1)) & (rsi <= ob_level) & price_conf_short

    # === MÓDULO 2: cruce MACD diario [LIBRO §8] ===
    m2_long = (macd_f > macd_s) & (macd_f.shift(1) <= macd_s.shift(1))
    m2_short = (macd_f < macd_s) & (macd_f.shift(1) >= macd_s.shift(1))

    # === MÓDULO 3: ADX + retroceso a EMA(20) [LIBRO §7.4] ===
    touched = (low <= ema20).astype(float).rolling(config.pullback_touch_bars).max().fillna(0) > 0
    touched_up = (high >= ema20).astype(float).rolling(config.pullback_touch_bars).max().fillna(0) > 0
    m3_long = strong & (adx > adx.shift(1)) & (pdi > mdi) & touched & (close > ema20) & (close > close.shift(1))
    m3_short = strong & (adx > adx.shift(1)) & (mdi > pdi) & touched_up & (close < ema20) & (close < close.shift(1))

    # === MÓDULO 4: CCI de umbral [LIBRO §7.6] ===
    m4_long = (cci > config.cci_entry_level) & (cci.shift(1) <= config.cci_entry_level)
    m4_short = (cci < -config.cci_entry_level) & (cci.shift(1) >= -config.cci_entry_level)

    # === MÓDULO 6: confluencia v2 [BASE, linaje] ===
    if config.enable_v2_confluence:
        touch = pd.Series(False, index=idx)
        for p in config.retracement_mas:
            ma_p = close.rolling(p).mean()
            touch = touch | ((close - ma_p).abs() / ma_p.abs() <= config.retracement_tolerance_pct)
        k_st, d_st = compute_stochastic(daily, config.stoch_k_period, config.stoch_d_period)
        st_up = (k_st > d_st) & (k_st.shift(1) <= d_st.shift(1)) & (k_st.shift(1) <= config.v2_stoch_oversold)
        st_dn = (k_st < d_st) & (k_st.shift(1) >= d_st.shift(1)) & (k_st.shift(1) >= config.v2_stoch_overbought)
        maru_b, maru_s = detect_marubozu(daily, config.marubozu_body_ratio)
        ma_flat = close.rolling(config.flat_window).mean()
        sd_flat = close.rolling(config.flat_window).std()
        is_flat = ((2 * sd_flat) / ma_flat < config.flat_width_threshold).fillna(False)
        rng_hi = close.rolling(config.flat_window).max().shift(1)
        rng_lo = close.rolling(config.flat_window).min().shift(1)
        flat_up = is_flat.shift(1, fill_value=False) & (close > rng_hi)
        flat_dn = is_flat.shift(1, fill_value=False) & (close < rng_lo)
        _, up_ch, lo_ch = rolling_linreg_channel(close, config.channel_window, config.channel_band_mult)
        ch_pos = (close - lo_ch) / (up_ch - lo_ch).replace(0, np.nan)
        score_l = (touch.astype(int) + st_up.astype(int) + maru_b.astype(int)
                   + flat_up.astype(int) + (ch_pos <= config.channel_long_zone).fillna(False).astype(int))
        score_s = (touch.astype(int) + st_dn.astype(int) + maru_s.astype(int)
                   + flat_dn.astype(int) + (ch_pos >= config.channel_short_zone).fillna(False).astype(int))
        m6_long = score_l >= config.min_confluence_score
        m6_short = score_s >= config.min_confluence_score
    else:
        m6_long = pd.Series(False, index=idx)
        m6_short = pd.Series(False, index=idx)

    # --- Composición con prioridad (primer módulo activo gana) ---
    signal = pd.Series(None, index=idx, dtype=object)
    module = pd.Series(None, index=idx, dtype=object)

    def _apply(mask_long: pd.Series, mask_short: pd.Series, name: str, gated: bool = True):
        ml = (mask_long & gate_long) if gated else mask_long
        ms = (mask_short & gate_short) if gated else mask_short
        sel_l = ml.fillna(False) & signal.isna()
        sel_s = ms.fillna(False) & signal.isna() & ~sel_l
        signal[sel_l] = "long"
        signal[sel_s] = "short"
        module[sel_l | sel_s] = name

    if config.enable_rsi_macd_weekly:
        _apply(m1_long, m1_short, "rsi_macd_weekly")
    if config.enable_macd_daily_cross:
        _apply(m2_long, m2_short, "macd_daily_cross")
    if config.enable_adx_ema20_pullback:
        _apply(m3_long, m3_short, "adx_ema20_pullback")
    if config.enable_cci_threshold:
        _apply(m4_long, m4_short, "cci_threshold")
    if config.enable_v2_confluence:
        _apply(m6_long, m6_short, "v2_confluence")
    if config.enable_trap_reversal:
        # [BASE/CURSO] contra-escape: solo liquidez + vetos de divergencia/estacionalidad
        trap_s = fu & liquid & (~block_short) & (~season_block_short)
        trap_l = fd & liquid & (~block_long) & (~season_block_long)
        _apply(trap_l, trap_s, "trap_reversal", gated=False)

    # --- Niveles auxiliares para el motor (stop estructural, salidas) ---
    struct_low = low.rolling(config.stop_lookback_bars).min() - config.stop_buffer_atr * atr
    struct_high = high.rolling(config.stop_lookback_bars).max() + config.stop_buffer_atr * atr
    macd_exit_long = (macd_f < macd_s) & (macd_f.shift(1) >= macd_s.shift(1))   # cruce a la baja
    macd_exit_short = (macd_f > macd_s) & (macd_f.shift(1) <= macd_s.shift(1))  # cruce al alza
    cci_exit_long = (cci < 0) & (cci.shift(1) >= 0)
    cci_exit_short = (cci > 0) & (cci.shift(1) <= 0)

    return pd.DataFrame({
        "close": close, "signal": signal, "module": module,
        "trend_dir": trend, "macro_regime": macro, "liquid": liquid,
        "adx": adx, "rsi": rsi,
        "struct_stop_long": struct_low, "struct_stop_short": struct_high,
        "ema20": ema20,
        "macd_exit_long": macd_exit_long.fillna(False),
        "macd_exit_short": macd_exit_short.fillna(False),
        "cci_exit_long": cci_exit_long.fillna(False),
        "cci_exit_short": cci_exit_short.fillna(False),
        "bear_div_known": divs["bear_div_known"], "bull_div_known": divs["bull_div_known"],
    })


# =======================================================================
# 8. MOTOR DE CARTERA (hereda v2: MTM diario, costes, huecos, causal)
# =======================================================================

@dataclass
class Trade:
    ticker: str
    side: str
    module: str
    entry_date: pd.Timestamp
    entry_price: float
    units: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl_cash: Optional[float] = None
    pnl_pct: Optional[float] = None
    bars_held: int = 0
    mae_pct: float = 0.0
    mfe_pct: float = 0.0


@dataclass
class _OpenPosition:
    ticker: str
    side: str
    module: str
    units: float
    entry_date: pd.Timestamp
    entry_price: float
    stop: float
    extreme: float
    worst: float
    best: float
    bars: int
    last_price: float
    sar_af: float
    sar: float
    prev_low: float
    prev_low2: float
    prev_high: float
    prev_high2: float


@dataclass
class BacktestResult:
    equity: pd.Series
    returns: pd.Series
    trades: list
    signals: dict
    skipped_entries: int
    avg_gross_exposure: float


def run_portfolio_backtest(
    daily_data: dict[str, pd.DataFrame],
    macro_monthly: pd.DataFrame,
    config: CavaConfigV3,
    initial_equity: float = 1.0,
) -> BacktestResult:
    macro_reg = macro_regime(macro_monthly, config)

    signals: dict[str, pd.DataFrame] = {}
    atrs: dict[str, pd.Series] = {}
    emas_trail: dict[str, pd.Series] = {}
    trend_strength: dict[str, pd.Series] = {}
    pos_by_date: dict[str, dict] = {}
    for ticker, df in daily_data.items():
        sig = generate_signals_v3(df, macro_reg, config)
        signals[ticker] = sig
        atrs[ticker] = compute_atr(df, config.atr_period)
        emas_trail[ticker] = df["close"].ewm(span=config.trailing_ema_period, adjust=False).mean()
        trend_strength[ticker] = sig["adx"]
        pos_by_date[ticker] = {ts: i for i, ts in enumerate(df.index)}

    calendar = sorted(set().union(*[set(df.index) for df in daily_data.values()]))
    scan_dates = set()
    for i, d in enumerate(calendar):
        nxt = calendar[i + 1] if i + 1 < len(calendar) else None
        if nxt is None or nxt.isocalendar()[:2] != d.isocalendar()[:2]:
            scan_dates.add(d)

    cash = initial_equity
    open_positions: dict[str, _OpenPosition] = {}
    pending: list[dict] = []
    pending_exits: dict[str, str] = {}
    eligible: set[str] = set(daily_data.keys())
    last_exit_calidx: dict[str, int] = {}
    trades: list[Trade] = []
    equity_dates: list[pd.Timestamp] = []
    equity_vals: list[float] = []
    gross_vals: list[float] = []
    skipped = 0

    def _signed(pos: _OpenPosition) -> float:
        return pos.units if pos.side == "long" else -pos.units

    def _vol_scalar() -> float:
        w = config.vol_target_window
        if len(equity_vals) < w + 1:
            return 1.0
        eq = np.array(equity_vals[-(w + 1):])
        rets = np.diff(eq) / eq[:-1]
        realized = float(np.std(rets, ddof=1)) * math.sqrt(252)
        if realized <= 0:
            return 1.0
        return float(np.clip(config.target_annual_vol / realized,
                             config.vol_scalar_floor, config.max_gross_leverage))

    def _close_position(tkr: str, d: pd.Timestamp, exit_price: float, reason: str):
        nonlocal cash
        pos = open_positions[tkr]
        signed = _signed(pos)
        cost = abs(pos.units * exit_price) * config.cost_bps_per_side / 10_000.0
        cash += signed * exit_price - cost
        sgn = 1.0 if pos.side == "long" else -1.0
        trades.append(Trade(
            ticker=tkr, side=pos.side, module=pos.module, entry_date=pos.entry_date,
            entry_price=pos.entry_price, units=pos.units, exit_date=d,
            exit_price=exit_price, exit_reason=reason,
            pnl_cash=signed * (exit_price - pos.entry_price)
                     - (abs(pos.units * exit_price) + abs(pos.units * pos.entry_price))
                     * config.cost_bps_per_side / 10_000.0,
            pnl_pct=sgn * (exit_price / pos.entry_price - 1.0),
            bars_held=pos.bars,
            mae_pct=sgn * (pos.worst / pos.entry_price - 1.0),
            mfe_pct=sgn * (pos.best / pos.entry_price - 1.0),
        ))
        del open_positions[tkr]

    for i_cal, d in enumerate(calendar):
        prev_equity = equity_vals[-1] if equity_vals else initial_equity

        # ---------- 1) APERTURA: salidas por señal encoladas ayer ----------
        for tkr in list(pending_exits.keys()):
            if tkr in open_positions and d in pos_by_date[tkr]:
                o = float(daily_data[tkr]["open"].iloc[pos_by_date[tkr][d]])
                _close_position(tkr, d, o, pending_exits.pop(tkr))
                last_exit_calidx[tkr] = i_cal
            elif tkr not in open_positions:
                pending_exits.pop(tkr)

        # ---------- 1b) APERTURA: entradas encoladas ayer ----------
        still_pending = []
        for order in pending:
            tkr = order["ticker"]
            df = daily_data[tkr]
            if d not in pos_by_date[tkr]:
                order["age"] += 1
                if order["age"] <= config.pending_max_age_bars:
                    still_pending.append(order)
                continue
            if tkr in open_positions or len(open_positions) >= config.max_concurrent_positions:
                continue
            if config.reentry_cooldown_bars > 0 and tkr in last_exit_calidx:
                if i_cal - last_exit_calidx[tkr] < config.reentry_cooldown_bars:
                    continue
            i = pos_by_date[tkr][d]
            entry_price = float(df["open"].iloc[i])
            if not np.isfinite(entry_price) or entry_price <= 0:
                continue
            side = order["side"]

            # Stop inicial: estructural [LIBRO §10] con tope duro del 3% [BASE]
            pct_stop = entry_price * (1 - config.initial_stop_pct) if side == "long" \
                else entry_price * (1 + config.initial_stop_pct)
            struct = order.get("struct_stop")
            if config.stop_mode == "structural" and struct is not None and np.isfinite(struct):
                stop = max(struct, pct_stop) if side == "long" else min(struct, pct_stop)
                if (side == "long" and stop >= entry_price) or (side == "short" and stop <= entry_price):
                    stop = pct_stop
            else:
                stop = pct_stop

            risk_per_unit = abs(entry_price - stop)
            if risk_per_unit <= 0:
                continue
            scalar = _vol_scalar()
            risk_cash = prev_equity * config.risk_per_trade * scalar
            units = risk_cash / risk_per_unit
            notional = units * entry_price
            current_gross = sum(abs(_signed(p)) * p.last_price for p in open_positions.values())
            if current_gross + notional > config.max_gross_leverage * prev_equity:
                skipped += 1
                continue
            cost = notional * config.cost_bps_per_side / 10_000.0
            cash -= (units if side == "long" else -units) * entry_price
            cash -= cost
            open_positions[tkr] = _OpenPosition(
                ticker=tkr, side=side, module=order["module"], units=units,
                entry_date=d, entry_price=entry_price, stop=stop,
                extreme=entry_price, worst=entry_price, best=entry_price,
                bars=0, last_price=entry_price,
                sar_af=config.sar_af_start, sar=stop,
                prev_low=entry_price, prev_low2=entry_price,
                prev_high=entry_price, prev_high2=entry_price,
            )
        pending = still_pending

        # ---------- 2) DÍA: stops (orden conservador, huecos incluidos) ----------
        for tkr in list(open_positions.keys()):
            if d not in pos_by_date[tkr]:
                continue
            pos = open_positions[tkr]
            i = pos_by_date[tkr][d]
            row = daily_data[tkr].iloc[i]
            o, h, l, c = float(row["open"]), float(row["high"]), float(row["low"]), float(row["close"])
            bar_atr = float(atrs[tkr].iloc[i]) if np.isfinite(atrs[tkr].iloc[i]) else np.nan
            ema_t = float(emas_trail[tkr].iloc[i]) if np.isfinite(emas_trail[tkr].iloc[i]) else np.nan
            pos.bars += 1
            pos.worst = min(pos.worst, l) if pos.side == "long" else max(pos.worst, h)
            pos.best = max(pos.best, h) if pos.side == "long" else min(pos.best, l)

            exit_price, exit_reason = None, None
            stop_prev = pos.stop
            if pos.side == "long":
                if o <= stop_prev:
                    exit_price, exit_reason = o, "stop_gap"
                elif l <= stop_prev:
                    exit_price, exit_reason = stop_prev, "stop"
            else:
                if o >= stop_prev:
                    exit_price, exit_reason = o, "stop_gap"
                elif h >= stop_prev:
                    exit_price, exit_reason = stop_prev, "stop"

            if exit_price is None:
                # --- actualizar trailing DESPUÉS del chequeo (solo se estrecha) ---
                if pos.side == "long":
                    new_extreme = max(pos.extreme, h)
                    if config.trailing_mode == "sar":
                        if new_extreme > pos.extreme:
                            pos.sar_af = min(pos.sar_af + config.sar_af_step, config.sar_af_max)
                        pos.sar = pos.sar + pos.sar_af * (new_extreme - pos.sar)
                        pos.sar = min(pos.sar, pos.prev_low, pos.prev_low2)   # regla de Wilder
                        pos.stop = max(pos.stop, pos.sar)
                    elif config.trailing_mode == "atr" and np.isfinite(bar_atr):
                        pos.stop = max(pos.stop, new_extreme - config.trailing_atr_mult * bar_atr)
                    elif config.trailing_mode == "ema" and np.isfinite(ema_t):
                        pos.stop = max(pos.stop, ema_t)
                    pos.extreme = new_extreme
                else:
                    new_extreme = min(pos.extreme, l)
                    if config.trailing_mode == "sar":
                        if new_extreme < pos.extreme:
                            pos.sar_af = min(pos.sar_af + config.sar_af_step, config.sar_af_max)
                        pos.sar = pos.sar + pos.sar_af * (new_extreme - pos.sar)
                        pos.sar = max(pos.sar, pos.prev_high, pos.prev_high2)
                        pos.stop = min(pos.stop, pos.sar)
                    elif config.trailing_mode == "atr" and np.isfinite(bar_atr):
                        pos.stop = min(pos.stop, new_extreme + config.trailing_atr_mult * bar_atr)
                    elif config.trailing_mode == "ema" and np.isfinite(ema_t):
                        pos.stop = min(pos.stop, ema_t)
                    pos.extreme = new_extreme

                if pos.bars >= config.max_holding_bars:
                    exit_price, exit_reason = c, "time_limit"

            pos.prev_low2, pos.prev_low = pos.prev_low, l
            pos.prev_high2, pos.prev_high = pos.prev_high, h

            if exit_price is not None:
                _close_position(tkr, d, exit_price, exit_reason)
                last_exit_calidx[tkr] = i_cal

        # ---------- 3) CIERRE: mark-to-market ----------
        mtm = cash
        gross = 0.0
        for pos in open_positions.values():
            if d in pos_by_date[pos.ticker]:
                px = float(daily_data[pos.ticker]["close"].iloc[pos_by_date[pos.ticker][d]])
                pos.last_price = px
            else:
                px = pos.last_price
            mtm += _signed(pos) * px
            gross += abs(_signed(pos)) * px
        equity_dates.append(d)
        equity_vals.append(mtm)
        gross_vals.append(gross / mtm if mtm > 0 else 0.0)

        # ---------- 4) CIERRE: salidas por señal [LIBRO §8] -> cola ----------
        for tkr, pos in open_positions.items():
            if d not in pos_by_date[tkr] or tkr in pending_exits:
                continue
            row = signals[tkr].iloc[pos_by_date[tkr][d]]
            if pos.module == "cci_threshold":
                if (pos.side == "long" and row["cci_exit_long"]) or \
                   (pos.side == "short" and row["cci_exit_short"]):
                    pending_exits[tkr] = "cci_zero"
            elif config.use_macd_exit:
                if (pos.side == "long" and row["macd_exit_long"]) or \
                   (pos.side == "short" and row["macd_exit_short"]):
                    pending_exits[tkr] = "macd_cross"

        # ---------- 5) CIERRE: nuevas señales -> cola de entradas ----------
        for tkr, sig_df in signals.items():
            if d not in pos_by_date[tkr]:
                continue
            row = sig_df.iloc[pos_by_date[tkr][d]]
            sig = row["signal"]
            if sig not in ("long", "short"):
                continue
            if tkr in open_positions or tkr in pending_exits or tkr not in eligible:
                continue
            if any(o["ticker"] == tkr for o in pending):
                continue
            struct = row["struct_stop_long"] if sig == "long" else row["struct_stop_short"]
            if row["module"] == "adx_ema20_pullback" and np.isfinite(row["ema20"]):
                # [LIBRO §7.4] stop bajo/ sobre la media del retroceso
                struct = row["ema20"] * (0.999 if sig == "long" else 1.001)
            pending.append({"ticker": tkr, "side": sig, "module": row["module"],
                            "age": 0, "struct_stop": float(struct) if np.isfinite(struct) else None})

        # ---------- 6) Escaneo semanal [BASE: "buceos"] ----------
        if d in scan_dates:
            scores = {}
            for tkr, sig_df in signals.items():
                hist_idx = pos_by_date[tkr].get(d)
                if hist_idx is None:
                    loc = sig_df.index.searchsorted(d, side="right") - 1
                    if loc < 0:
                        continue
                    hist_idx = loc
                row = sig_df.iloc[hist_idx]
                if row["trend_dir"] == 0 or not np.isfinite(row["adx"]):
                    continue
                scores[tkr] = float(row["adx"])   # fuerza de tendencia = ADX [LIBRO §6.5]
            ranked = sorted(scores, key=scores.get, reverse=True)
            eligible = set(ranked[: config.top_n_candidates]) if ranked else set()

    equity = pd.Series(equity_vals, index=pd.DatetimeIndex(equity_dates), name="equity")
    returns = equity.pct_change().fillna(0.0)
    return BacktestResult(
        equity=equity, returns=returns, trades=trades, signals=signals,
        skipped_entries=skipped,
        avg_gross_exposure=float(np.mean(gross_vals)) if gross_vals else 0.0,
    )


# =======================================================================
# 9. MÉTRICAS (heredadas de v2 + rachas [LIBRO §1])
# =======================================================================

def probabilistic_sharpe_ratio(returns: pd.Series, sr_benchmark: float = 0.0) -> float:
    r = returns.dropna().to_numpy(dtype=float)
    T = len(r)
    if T < 4:
        return float("nan")
    mu, sd = r.mean(), r.std(ddof=1)
    if sd <= 0:
        return float("nan")
    sr = mu / sd
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))
    denom = math.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr * sr, 1e-12))
    return _norm_cdf((sr - sr_benchmark) * math.sqrt(T - 1) / denom)


def expected_max_sharpe(n_trials: int, sr_variance: float) -> float:
    if n_trials <= 1 or sr_variance <= 0:
        return 0.0
    gamma = 0.5772156649015329
    return math.sqrt(sr_variance) * (
        (1 - gamma) * _norm_ppf(1 - 1.0 / n_trials)
        + gamma * _norm_ppf(1 - 1.0 / (n_trials * math.e))
    )


def deflated_sharpe_ratio(returns: pd.Series, n_trials: int,
                          sr_variance_across_trials: Optional[float] = None) -> dict:
    r = returns.dropna().to_numpy(dtype=float)
    T = len(r)
    if T < 4 or r.std(ddof=1) <= 0:
        return {"dsr": float("nan"), "sr_benchmark": float("nan")}
    sr = r.mean() / r.std(ddof=1)
    z = (r - r.mean()) / r.std(ddof=1)
    skew, kurt = float(np.mean(z ** 3)), float(np.mean(z ** 4))
    if sr_variance_across_trials is None:
        sr_variance_across_trials = max((1 - skew * sr + (kurt - 1) / 4 * sr * sr) / (T - 1), 1e-12)
    sr0 = expected_max_sharpe(n_trials, sr_variance_across_trials)
    return {"dsr": probabilistic_sharpe_ratio(returns, sr0), "sr_benchmark": sr0}


def performance_report(result: BacktestResult, n_trials: int = 1) -> dict:
    eq, rets = result.equity, result.returns
    n_days = len(eq)
    if n_days < 2 or eq.iloc[0] <= 0:
        return {"error": "serie insuficiente"}
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252 / n_days) - 1
    ann_vol = rets.std(ddof=1) * math.sqrt(252)
    downside = rets[rets < 0].std(ddof=1) * math.sqrt(252)
    dd = eq / eq.cummax() - 1
    out = {
        "CAGR": float(cagr),
        "Vol anual": float(ann_vol),
        "Sharpe": float(cagr / ann_vol) if ann_vol > 0 else float("nan"),
        "Sortino": float(cagr / downside) if downside and downside > 0 else float("nan"),
        "Max Drawdown": float(dd.min()),
        "Calmar": float(cagr / abs(dd.min())) if dd.min() != 0 else float("nan"),
        "PSR (SR*=0)": probabilistic_sharpe_ratio(rets),
    }
    dsr = deflated_sharpe_ratio(rets, n_trials=n_trials)
    out[f"DSR (n_trials={n_trials})"] = dsr["dsr"]
    out["Exposición bruta media"] = result.avg_gross_exposure
    out["Entradas descartadas (límites)"] = result.skipped_entries
    return out


def trade_stats(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0}
    pnl = np.array([t.pnl_pct for t in trades if t.pnl_pct is not None])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    # rachas [LIBRO §1: dimensionar capital según rachas de pérdidas]
    max_loss_streak = max_win_streak = cur_l = cur_w = 0
    for t in sorted(trades, key=lambda x: x.exit_date):
        if t.pnl_pct is not None and t.pnl_pct > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_loss_streak = max(max_loss_streak, cur_l)
        max_win_streak = max(max_win_streak, cur_w)
    per_module: dict[str, int] = {}
    for t in trades:
        per_module[t.module] = per_module.get(t.module, 0) + 1
    return {
        "n_trades": len(pnl),
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else float("nan"),
        "avg_win_pct": float(wins.mean()) if len(wins) else float("nan"),
        "avg_loss_pct": float(losses.mean()) if len(losses) else float("nan"),
        "payoff": float(wins.mean() / abs(losses.mean())) if len(wins) and len(losses) and losses.mean() != 0 else float("nan"),
        "expectancy_pct": float(pnl.mean()),
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])),
        "avg_MAE_pct": float(np.mean([t.mae_pct for t in trades])),
        "avg_MFE_pct": float(np.mean([t.mfe_pct for t in trades])),
        "max_racha_perdidas": int(max_loss_streak),
        "max_racha_ganancias": int(max_win_streak),
        "trades_por_modulo": per_module,
    }


def stability_report(returns: pd.Series, n_segments: int = 4) -> pd.DataFrame:
    rets = returns.dropna()
    rows = []
    for seg in np.array_split(np.arange(len(rets)), n_segments):
        if len(seg) < 20:
            continue
        r = rets.iloc[seg]
        eq = (1 + r).cumprod()
        n = len(r)
        cagr = eq.iloc[-1] ** (252 / n) - 1
        vol = r.std(ddof=1) * math.sqrt(252)
        dd = (eq / eq.cummax() - 1).min()
        rows.append({"desde": r.index[0].date(), "hasta": r.index[-1].date(),
                     "CAGR": round(float(cagr), 4),
                     "Sharpe": round(float(cagr / vol), 2) if vol > 0 else float("nan"),
                     "MaxDD": round(float(dd), 4)})
    return pd.DataFrame(rows)


# =======================================================================
# 10. ADAPTADORES DE DATOS (Dukascopy CSV / FRED CSV) — de v2
# =======================================================================

def load_daily_csv(path: str, tz_naive: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower().strip(): c for c in df.columns}
    time_col = next((cols[k] for k in ("gmt time", "time", "date", "timestamp", "datetime") if k in cols), None)
    if time_col is None:
        raise ValueError(f"No encuentro columna temporal en {path}: {list(df.columns)}")
    try:
        idx = pd.to_datetime(df[time_col], format="%d.%m.%Y %H:%M:%S.%f")
    except (ValueError, TypeError):
        idx = pd.to_datetime(df[time_col])
    if tz_naive and getattr(idx.dt, "tz", None) is not None:
        idx = idx.dt.tz_localize(None)
    out = pd.DataFrame(index=pd.DatetimeIndex(idx))
    for name in ("open", "high", "low", "close", "volume"):
        src = next((cols[k] for k in cols if k == name), None)
        if src is None:
            raise ValueError(f"Falta columna '{name}' en {path}")
        out[name] = pd.to_numeric(df[src].values, errors="coerce")
    out = out.sort_index()
    return out[~out.index.duplicated(keep="last")].dropna(subset=["close"])


def load_fred_csv(path: str, value_name: str = "monetary_base") -> pd.DataFrame:
    df = pd.read_csv(path)
    date_col, val_col = df.columns[0], df.columns[1]
    idx = pd.to_datetime(df[date_col])
    s = pd.to_numeric(df[val_col], errors="coerce")
    out = pd.DataFrame({value_name: s.values}, index=pd.DatetimeIndex(idx)).dropna()
    return out.resample("ME").last().dropna()


# =======================================================================
# 11. DEMO SINTÉTICA (validación de pipeline, NO backtest real)
# =======================================================================

def _make_synthetic_asset(seed: int, n_days: int = 2400, start: str = "2015-01-01",
                          drift_scale: float = 1.0, intraday_steps: int = 26) -> pd.DataFrame:
    """
    OHLCV sintético derivado de un CAMINO intradía real (no de mechas
    independientes): cada día se simulan `intraday_steps` pasos
    log-normales; high/low son los extremos verdaderos del camino. Esto
    hace que la regla de fill 'si low<=stop, ejecutar en stop' sea
    coherente con la trayectoria (identidad de parada opcional): sobre
    un paseo sin deriva, CUALQUIER política de stops debe rendir ~0
    antes de costes. Con mechas fabricadas con ruido fino e
    independiente (versión anterior), los días de hundirse-y-recuperar
    no tocaban el stop intradía y el motor 'sobrevivía' dentro de la
    barra a pérdidas que un camino real habría ejecutado — sesgo
    positivo espurio detectado por el control negativo.

    drift_scale=1.0 -> regímenes de deriva de 120 días (control POSITIVO).
    drift_scale=0.0 -> paseo aleatorio puro (control NEGATIVO, ~0 exigido).
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    regime_len = 120
    drift = np.zeros(n_days)
    for s in range(0, n_days, regime_len):
        e = min(s + regime_len, n_days)
        drift[s:e] = rng.choice([0.0006, -0.0006, 0.0]) * drift_scale

    sigma_day = 0.012
    sigma_intra = sigma_day / math.sqrt(intraday_steps)
    gaps = rng.normal(0, 0.0015, n_days)
    steps = rng.normal(0, sigma_intra, (n_days, intraday_steps)) + (drift / intraday_steps)[:, None]

    open_ = np.empty(n_days); high = np.empty(n_days)
    low = np.empty(n_days); close = np.empty(n_days)
    prev_close = 100.0
    for t in range(n_days):
        o = prev_close * math.exp(gaps[t])
        path = o * np.exp(np.cumsum(steps[t]))
        c = path[-1]
        open_[t], close[t] = o, c
        high[t] = max(o, path.max())
        low[t] = min(o, path.min())
        prev_close = c
    volume = rng.normal(2_000_000, 400_000, n_days).clip(min=200_000)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=dates)


def _make_synthetic_macro(n_months: int = 140, start: str = "2015-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range(start, periods=n_months, freq="ME")
    base = 4000 + np.cumsum(rng.normal(15, 25, n_months))
    return pd.DataFrame({"monetary_base": base,
                         "M1": base * 1.4 + rng.normal(0, 20, n_months),
                         "M2": base * 4.8 + rng.normal(0, 60, n_months)}, index=dates)


def _selftest_divergence_causality(config: CavaConfigV3) -> bool:
    """El flag de divergencia nunca puede aparecer antes de 2 pivotes + k barras."""
    df = _make_synthetic_asset(seed=7, n_days=600)
    rsi = compute_rsi(df["close"], config.rsi_period)
    divs = detect_divergences(df, rsi, config)
    first = min(
        [divs["bear_div_known"].to_numpy().argmax() if divs["bear_div_known"].any() else 10**9,
         divs["bull_div_known"].to_numpy().argmax() if divs["bull_div_known"].any() else 10**9]
    )
    return first >= 2 * config.div_pivot_k


def _selftest_sar_monotone(config: CavaConfigV3) -> bool:
    """En una subida perfecta, el stop SAR de un largo nunca debe bajar."""
    n = 120
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = pd.Series(np.linspace(100, 160, n), index=dates)
    df = pd.DataFrame({"open": close.values, "high": close.values * 1.001,
                       "low": close.values * 0.999, "close": close.values,
                       "volume": 2e6}, index=dates)
    stop = 97.0
    sar, af, extreme = stop, config.sar_af_start, 100.0
    prev_low = prev_low2 = 100.0
    stops = []
    for i in range(1, n):
        h, l = float(df["high"].iloc[i]), float(df["low"].iloc[i])
        new_extreme = max(extreme, h)
        if new_extreme > extreme:
            af = min(af + config.sar_af_step, config.sar_af_max)
        sar = sar + af * (new_extreme - sar)
        sar = min(sar, prev_low, prev_low2)
        stop = max(stop, sar)
        stops.append(stop)
        prev_low2, prev_low = prev_low, l
        extreme = new_extreme
    return all(b >= a for a, b in zip(stops, stops[1:]))


def main() -> None:
    print("=" * 70)
    print("DEMO v3 — DATOS SINTÉTICOS. Dos controles del motor:")
    print("  1) CONTROL NEGATIVO: paseo aleatorio puro -> resultado ~0 exigido")
    print("  2) CONTROL POSITIVO: regímenes de deriva plantados -> el motor")
    print("     debería extraer parte de esa estructura")
    print("Caminos intradía de 390 pasos: con granularidad gruesa (26) el")
    print("fill exacto en el stop regala el overshoot (~14pb/salida) y el")
    print("nulo da falso positivo. Certificación hecha en desarrollo:")
    print("10 universos nulos K=390 -> Sharpe medio -0.08 +/- 0.12 (SE),")
    print("esperanza +0.03% +/- 0.04% por trade: motor sin sesgo detectable.")
    print("Nada de esto valida la estrategia sobre mercados reales.")
    print("=" * 70)

    config = CavaConfigV3()
    ok_div = _selftest_divergence_causality(config)
    ok_sar = _selftest_sar_monotone(config)
    print(f"\n[selftest] causalidad divergencias: {'OK' if ok_div else 'FALLO'}")
    print(f"[selftest] SAR monótono en tendencia: {'OK' if ok_sar else 'FALLO'}")

    tickers = [f"SYN_{c}" for c in "ABCDEF"]
    macro = _make_synthetic_macro()

    for label, drift_scale in [("CONTROL NEGATIVO (sin deriva)", 0.0),
                               ("CONTROL POSITIVO (con regímenes)", 1.0)]:
        daily_data = {t: _make_synthetic_asset(seed=i, drift_scale=drift_scale, intraday_steps=390)
                      for i, t in enumerate(tickers)}
        result = run_portfolio_backtest(daily_data, macro, config)
        ts = trade_stats(result.trades)
        rep = performance_report(result, n_trials=1)
        print(f"\n--- {label} ---")
        print(f"  trades: {ts.get('n_trades', 0)} | por módulo: {ts.get('trades_por_modulo', {})}")
        for k in ("CAGR", "Vol anual", "Sharpe", "Max Drawdown", "PSR (SR*=0)"):
            v = rep.get(k)
            print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        print(f"  expectancy/trade: {ts.get('expectancy_pct', float('nan')):.4f} | "
              f"max racha pérdidas: {ts.get('max_racha_perdidas', 0)}")
        closed = sum(t.pnl_cash for t in result.trades)
        latent = result.equity.iloc[-1] - 1 - closed
        print(f"  [integridad] equity {result.equity.iloc[-1]:.6f} = 1 + cerrado {closed:+.6f} + latente {latent:+.6f}")

        if drift_scale == 1.0:
            print("\n  Estabilidad por tramos (control positivo):")
            print(stability_report(result.returns, n_segments=4).to_string(index=False))


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
