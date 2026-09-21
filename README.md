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

### pinescript (documentación Pine Script v6)

Segundo servidor MCP, [pinescript-mcp-server](https://github.com/cklose2000/pinescript-mcp-server)
(paquete npm, se ejecuta con `npx`, sin instalación manual ni API key).
Da acceso a la documentación oficial de Pine Script v6: búsqueda, referencia
de funciones/variables, guías, ejemplos y categorías. Útil para trabajar con
`pine/cava_trend_strategy_v3.pine`.

Herramientas disponibles:
- `pine_search`: búsqueda de texto completo en la documentación.
- `pine_reference`: referencia exacta de una función/variable.
- `pine_guide`: guías de usuario por tema.
- `pine_examples`: ejemplos de código.
- `pine_categories`: categorías de funciones.
