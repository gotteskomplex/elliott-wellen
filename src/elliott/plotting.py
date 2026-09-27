"""Plotly charts. Pure presentation: reads analysis results, never computes counts."""

from __future__ import annotations

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots

from elliott.models import AnalysisResult, WaveCount
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
        height=cfg.height, xaxis_rangeslider_visible=False, hovermode="closest", template="plotly_white",
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


# ---------------------------------------------------------------------------
# analysis chart
# ---------------------------------------------------------------------------


def _hex_to_rgba(color: str, alpha: float) -> str:
    c = color.lstrip("#")
    r, g, b = int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _count_trace(count: WaveCount, color: str, name: str, width: float, dash: str, visible: bool | str,
                 font_size: int, legendgroup: str, showlegend: bool = True) -> tuple[go.Scatter, list[dict]]:
    xs = [count.waves[0].start.ts] + [w.end.ts for w in count.waves]
    ys = [count.waves[0].start.price] + [w.end.price for w in count.waves]
    hover = ["Start"] + [
        f"Welle {w.display_label}: {w.start.price:.4g} → {w.end.price:.4g}" + (" (läuft)" if w.open else "")
        for w in count.waves
    ]
    trace = go.Scatter(
        x=xs, y=ys, mode="lines+markers", name=name, legendgroup=legendgroup, showlegend=showlegend,
        line={"color": color, "width": width, "dash": dash}, marker={"size": 8, "color": color},
        hovertext=hover, hoverinfo="text", visible=visible,
    )
    labels = []
    for w in count.waves:
        up = w.end.kind.value == "high"
        labels.append({
            "x": w.end.ts, "y": w.end.price, "text": w.display_label, "showarrow": False,
            "yshift": 14 if up else -14, "font": {"size": font_size, "color": color},
        })
    return trace, labels


def plot_analysis(
    df: pd.DataFrame,
    result: AnalysisResult,
    cfg: PlottingConfig,
    scenario_ids: list[str] | None = None,
    log_axis: bool | None = None,
    show_volume: bool = True,
    show_subcounts: bool = True,
) -> go.Figure:
    """Candlestick chart with wave counts, target zones, invalidation and projection.

    The first selected scenario is the main scenario (bold); the others are
    added as toggleable traces (legend click).
    """
    log_axis = result.meta.log_scale if log_axis is None else log_axis
    fig = base_figure(df, cfg, log_axis=log_axis, show_volume=show_volume)
    scenarios = result.scenarios
    if scenario_ids is not None:
        order = {sid: i for i, sid in enumerate(scenario_ids)}
        scenarios = sorted((s for s in scenarios if s.id in order), key=lambda s: order[s.id])
    if not scenarios:
        fig.update_layout(title="Keine Szenarien gefunden")
        return fig

    step = df.index.to_series().diff().median() if len(df) > 1 else pd.Timedelta(days=1)
    future_bars = max(cfg.future_bars_min, int(len(df) * cfg.future_bars_ratio))
    x_end = df.index[-1] + step * future_bars
    main_annotations: list[dict] = []
    for i, sc in enumerate(scenarios):
        color = cfg.scenario_colors[(sc.rank - 1) % len(cfg.scenario_colors)]
        main = i == 0
        name = f"{sc.id}: {sc.count.pattern.german} ({sc.score.total:.0f})"
        trace, labels = _count_trace(
            sc.count, color, name, width=3 if main else 1.6, dash="solid" if main else "dash",
            visible=True if main else "legendonly", font_size=cfg.label_font_sizes[0], legendgroup=sc.id,
        )
        fig.add_trace(trace, row=1, col=1)
        if main:
            main_annotations.extend(labels)
        else:
            # label-only trace so labels toggle together with the scenario
            fig.add_trace(go.Scatter(
                x=[a["x"] for a in labels], y=[a["y"] for a in labels], mode="text", text=[a["text"] for a in labels],
                textposition=["top center" if a["yshift"] > 0 else "bottom center" for a in labels],
                textfont={"color": color, "size": cfg.label_font_sizes[1]}, legendgroup=sc.id, showlegend=False,
                visible="legendonly", hoverinfo="skip",
            ), row=1, col=1)
        if main and show_subcounts:
            sub_color = _hex_to_rgba(color, cfg.subcount_opacity)
            for w in sc.count.waves:
                if w.subcount is None:
                    continue
                t, sub_labels = _count_trace(
                    w.subcount, sub_color, f"{sc.id} Subwellen", 1, "dot", True,
                    cfg.label_font_sizes[min(1, len(cfg.label_font_sizes) - 1)], legendgroup=f"{sc.id}-sub", showlegend=False,
                )
                fig.add_trace(t, row=1, col=1)
                main_annotations.extend(sub_labels[:-1])
            proj_sub = sc.projection.current_subcount if sc.projection else None
            last_wave = sc.count.waves[-1]
            if proj_sub is not None and proj_sub.waves and last_wave.subcount is None:
                t, sub_labels = _count_trace(
                    proj_sub, sub_color, f"{sc.id} aktuelle Subwelle", 1, "dot", True,
                    cfg.label_font_sizes[min(1, len(cfg.label_font_sizes) - 1)], legendgroup=f"{sc.id}-sub", showlegend=False,
                )
                fig.add_trace(t, row=1, col=1)
                end = last_wave.end
                main_annotations.extend(a for a in sub_labels if not (a["x"] == end.ts and a["y"] == end.price))
        proj = sc.projection
        if proj is None:
            continue
        vis = True if main else "legendonly"
        # target zones as filled rectangles (traces so they toggle with the scenario)
        for z in proj.targets:
            alpha = cfg.target_opacity * (1.6 if z.highlighted else 1.0)
            fig.add_trace(go.Scatter(
                x=[df.index[-1], x_end, x_end, df.index[-1], df.index[-1]],
                y=[z.low, z.low, z.high, z.high, z.low], fill="toself", mode="lines",
                line={"width": 0}, fillcolor=_hex_to_rgba(color, min(1.0, alpha)),
                legendgroup=sc.id, showlegend=False, visible=vis,
                hoverinfo="text", hovertext=f"Ziel {z.low:.4g}–{z.high:.4g}: " + ", ".join(z.sources)
                + (" (Konfluenz)" if z.highlighted else ""),
            ), row=1, col=1)
        inv = proj.invalidation
        fig.add_trace(go.Scatter(
            x=[df.index[max(0, len(df) - future_bars)], x_end], y=[inv.price, inv.price], mode="lines",
            line={"color": cfg.invalidation_color, "dash": "dash", "width": 1.6 if main else 1},
            legendgroup=sc.id, showlegend=False, visible=vis, hoverinfo="text",
            hovertext=f"Invalidierung {sc.id}: {inv.reason}",
        ), row=1, col=1)
        if proj.path:
            fig.add_trace(go.Scatter(
                x=[p[0] for p in proj.path], y=[p[1] for p in proj.path], mode="lines",
                line={"color": color, "dash": "dot", "width": 2 if main else 1},
                legendgroup=sc.id, showlegend=False, visible=vis, hoverinfo="text",
                hovertext=f"Projektion {sc.id}: {proj.next_move}",
            ), row=1, col=1)
    fig.update_layout(annotations=main_annotations, title=f"Elliott-Wellen – Hauptszenario {scenarios[0].id}: "
                      f"{scenarios[0].projection.position if scenarios[0].projection else ''}")
    fig.update_xaxes(range=[df.index[0], x_end])
    return fig
