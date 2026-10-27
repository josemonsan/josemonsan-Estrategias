"""
Parámetros pre-registrados.

Congela este archivo (y GRID) ANTES de mirar resultados fuera de muestra.
Cada combinación del GRID y cada variante (con / sin on-chain) cuenta como
un trial para el Deflated Sharpe Ratio.
"""
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class StrategyParams:
    # --- Señal de tendencia (en barras del timeframe que cargues) ---
    sma_regime: int = 200          # régimen: largo solo sobre la SMA, corto solo debajo
    donchian_entry: int = 20       # ruptura del máximo/mínimo de las N barras previas
    donchian_exit: int = 20        # salida normal; con alerta on-chain se usa la mitad
    atr_len: int = 20              # N de las Tortugas (ATR de Wilder)

    # --- Riesgo y piramidación (Tortugas) ---
    stop_atr: float = 2.0          # stop común a 2N de la última entrada
    stop_atr_alert: float = 1.5    # stop ceñido cuando salta la alerta on-chain
    add_step_atr: float = 1.0      # añade una unidad cada 1N de avance
    max_units: int = 4
    risk_per_unit: float = 0.01    # % de la cuenta que se pierde por unidad si salta el stop
    max_leverage: float = 3.0      # techo de exposición nocional / equity

    allow_shorts: bool = True      # cortos permitidos en cripto
    use_onchain: bool = True       # variante B (con on-chain) vs A (sin)


@dataclass(frozen=True)
class OnchainParams:
    window_days: int = 7           # suma móvil de 7 días antes del z-score (reduce ruido)
    z_lookback_days: int = 365     # z-score causal contra el último año
    z_min_periods: int = 180
    alert_z: float = 1.5           # alerta si el compuesto supera este z
    confirm_days: int = 3          # alerta sostenida N días = señal confirmada -> salida total
    lag_hours: int = 6             # el dato del día D se usa desde D+1 06:00 UTC
    components: Tuple[str, ...] = ("whale_volume", "vdd", "exch_netflow")


@dataclass(frozen=True)
class CostParams:
    fee_rate: float = 0.001        # OKX perpetuos, 0,1% por lado
    slippage: float = 0.0005       # 5 pb por ejecución
    funding_rate_8h: float = 0.0001  # se usa solo si no cargas un CSV de funding


@dataclass(frozen=True)
class EvalParams:
    initial_equity: float = 10_000.0
    max_dd: float = 0.60           # criterio: drawdown máximo tolerado
    train_years: int = 3
    test_years: int = 1


# Rejilla pequeña a propósito: 8 combinaciones x 2 variantes = 16 trials.
GRID = {
    "donchian_entry": [20, 55],
    "donchian_exit": [10, 20],
    "sma_regime": [100, 200],
}
