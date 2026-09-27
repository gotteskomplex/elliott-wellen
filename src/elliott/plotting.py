"""Plotly charts. Pure presentation: reads analysis results, never computes counts."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

from elliott.pivots.zigzag import PivotHierarchy
from elliott.settings import PlottingConfig


def base_figure(df: pd.DataFrame, cfg: PlottingConfig, log_axis: bool = False, show_volume: bool = True) -> go.Figure:
    """Candlestick chart with optional volume subplot."""
    has_volume = show_volume and "volume" in df.columns
    if has_volume:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
            row_heights=[1.0 - cfg.volume_row_height, cfg.volume_row_height],
        )
    else:
        fig = make_subplots(rows=1, cols=1)
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
            name="Kurs", increasing_line_color=cfg.candle_up_color, decreasing_line_color=cfg.candle_down_color,
            showlegend=False,
        ),
        row=1, col=1,
    )
    if has_volume:
        colors = [cfg.candle_up_color if c >= o else cfg.candle_down_color for o, c in zip(df["open"], df["close"])]
        fig.add_trace(go.Bar(x=df.index, y=df["volume"], marker_color=colors, name="Volumen", showlegend=False), row=2, col=1)
    fig.update_layout(
        height=cfg.height, xaxis_rangeslider_visible=False, hovermode="x unified",
        margin={"l": 40, "r": 20, "t": 50, "b": 30}, legend={"orientation": "h", "y": 1.04},
    )
    fig.update_yaxes(type="log" if log_axis else "linear", row=1, col=1)
    return fig


def plot_pivots(df: pd.DataFrame, hierarchy: PivotHierarchy, cfg: PlottingConfig, log_axis: bool = False) -> go.Figure:
    """Control chart: ZigZag lines of all degrees on top of the candles."""
    fig = base_figure(df, cfg, log_axis=log_axis, show_volume=False)
    for level, pivots in enumerate(hierarchy.levels):
        if not pivots:
            continue
        color = cfg.pivot_colors[min(level, len(cfg.pivot_colors) - 1)]
        fig.add_trace(
            go.Scatter(
                x=[p.ts for p in pivots], y=[p.price for p in pivots], mode="lines+markers",
                line={"color": color, "width": max(1, 3 - level), "dash": "solid"},
                marker={"size": max(4, 9 - 2 * level), "symbol": ["circle-open" if p.provisional else "circle" for p in pivots]},
                name=f"{hierarchy.names[level]} ({hierarchy.thresholds[level].describe()}, {len(pivots)} Pivots)",
            )
        )
    fig.update_layout(title="Pivot-Kontrolle (ZigZag je Grad)")
    return fig
