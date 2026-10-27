"""Gráfico interactivo en HTML (Plotly): precio + operaciones, equity vs buy & hold,
drawdown y compuesto on-chain."""
from __future__ import annotations

import pandas as pd


def plot_html(df: pd.DataFrame, results: dict, bh: pd.Series, path: str,
              alert_z: float | None = None) -> None:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    has_oc = "oc_composite" in df
    rows = 4 if has_oc else 3
    heights = [0.45, 0.25, 0.15, 0.15] if has_oc else [0.5, 0.3, 0.2]
    titles = ["ETH", "Equity (log)", "Drawdown"] + (["Riesgo on-chain (z)"] if has_oc else [])
    fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, row_heights=heights,
                        vertical_spacing=0.03, subplot_titles=titles)

    fig.add_trace(go.Candlestick(x=df.index, open=df["open"], high=df["high"],
                                 low=df["low"], close=df["close"], name="ETH",
                                 showlegend=False), row=1, col=1)
    colors = {"A_sin_onchain": "#1f77b4", "B_con_onchain": "#d62728"}
    main = list(results)[-1]
    f = results[main].fills
    if not f.empty:
        buys, sells = f[f["side"] > 0], f[f["side"] < 0]
        fig.add_trace(go.Scatter(x=buys["time"], y=buys["price"], mode="markers",
                                 marker=dict(symbol="triangle-up", size=8, color="#2ca02c"),
                                 name=f"compras ({main})", text=buys["kind"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=sells["time"], y=sells["price"], mode="markers",
                                 marker=dict(symbol="triangle-down", size=8, color="#d62728"),
                                 name=f"ventas ({main})", text=sells["kind"]), row=1, col=1)

    for name, res in results.items():
        eq = res.equity
        fig.add_trace(go.Scatter(x=eq.index, y=eq, name=name,
                                 line=dict(color=colors.get(name))), row=2, col=1)
        dd = eq / eq.cummax() - 1
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=f"DD {name}", showlegend=False,
                                 line=dict(color=colors.get(name), width=1)), row=3, col=1)
    fig.add_trace(go.Scatter(x=bh.index, y=bh, name="buy & hold",
                             line=dict(color="#7f7f7f", dash="dot")), row=2, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)

    if has_oc:
        fig.add_trace(go.Scatter(x=df.index, y=df["oc_composite"], name="compuesto on-chain",
                                 line=dict(color="#9467bd", width=1)), row=4, col=1)
        if alert_z is not None:
            fig.add_hline(y=alert_z, line=dict(color="#d62728", dash="dash"), row=4, col=1)

    fig.update_layout(height=1100, xaxis_rangeslider_visible=False,
                      template="plotly_white", hovermode="x unified",
                      legend=dict(orientation="h", y=1.02))
    fig.write_html(path, include_plotlyjs="cdn")
