# Estrategias

## MCP de TradingView

Este repo incluye `.mcp.json` con la configuración del servidor MCP
[mcp-tradingview-server](https://github.com/bidouilles/mcp-tradingview-server),
que expone indicadores técnicos y datos OHLCV de TradingView.

Requiere tener [`uv`](https://docs.astral.sh/uv/) instalado (usa `uvx` para
ejecutar el servidor directamente desde el repositorio de GitHub, sin
instalación manual ni API key). Claude Code lo detecta automáticamente al
abrir este proyecto.

Herramientas disponibles:
- `get_indicators`: snapshot completo de indicadores para un símbolo/exchange/timeframe.
- `get_specific_indicators`: filtra indicadores por nombre.
- `get_historical_data`: velas OHLCV históricas.
