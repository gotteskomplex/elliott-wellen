from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from elliott.pivots.zigzag import bar_ticks, build_hierarchy, detect_pivots, zigzag_ticks
from elliott.plotting import plot_pivots
from elliott.settings import ThresholdSpec, with_overrides


def _bars(highs, lows, opens=None, closes=None) -> pd.DataFrame:
    n = len(highs)
    highs = np.asarray(highs, float)
    lows = np.asarray(lows, float)
    mid = (highs + lows) / 2
    opens = mid if opens is None else np.asarray(opens, float)
    closes = mid if closes is None else np.asarray(closes, float)
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes},
        index=pd.date_range("2024-01-01", periods=n, freq="D"),
    )


def _path(points: list[float], steps: int = 5) -> pd.DataFrame:
    """Candles following a piecewise linear path (tiny ranges)."""
    closes: list[float] = []
    for a, b in zip(points[:-1], points[1:]):
        closes.extend(np.linspace(a, b, steps, endpoint=False))
    closes.append(points[-1])
    c = np.asarray(closes)
    o = np.r_[c[0], c[:-1]]
    return _bars(np.maximum(o, c) + 0.01, np.minimum(o, c) - 0.01, o, c)


PCT5 = ThresholdSpec(mode="percent", value=5)


def _atr_const(n: int, value: float = 1.0) -> np.ndarray:
    return np.full(n, value)


def test_alternating_and_values() -> None:
    df = _path([100, 120, 110, 130, 115, 125])
    piv = detect_pivots(df, PCT5, _atr_const(len(df)))
    kinds = [p.kind.value for p in piv]
    assert kinds == ["low", "high", "low", "high", "low", "high"]
    assert [round(p.price) for p in piv] == [100, 120, 110, 130, 115, 125]
    assert all(a.kind != b.kind for a, b in zip(piv, piv[1:]))


def test_last_pivot_is_provisional() -> None:
    df = _path([100, 120, 110, 130, 127])
    piv = detect_pivots(df, PCT5, _atr_const(len(df)))
    assert piv[-1].provisional and round(piv[-1].price) == 130
    assert not any(p.provisional for p in piv[:-1])


def test_small_moves_ignored() -> None:
    df = _path([100, 120, 118, 121, 110])
    piv = detect_pivots(df, PCT5, _atr_const(len(df)))
    assert [round(p.price) for p in piv] == [100, 121, 110]


def test_atr_threshold() -> None:
    df = _path([100, 110, 107, 115, 104])
    spec = ThresholdSpec(mode="atr", value=2)
    piv = detect_pivots(df, spec, _atr_const(len(df), 2.0))  # 4 points required
    assert [round(p.price) for p in piv] == [100, 115, 104]
    piv_fine = detect_pivots(df, spec, _atr_const(len(df), 1.0))  # 2 points
    assert [round(p.price) for p in piv_fine] == [100, 110, 107, 115, 104]


def test_equal_highs_keep_first() -> None:
    df = _bars(highs=[101, 110, 105, 110, 104, 100], lows=[99, 105, 103, 104, 100, 95])
    piv = detect_pivots(df, PCT5, _atr_const(len(df)))
    highs = [p for p in piv if p.kind.value == "high"]
    assert highs[0].idx == 1


def test_outside_bar_bearish_creates_both_pivots() -> None:
    # Up-leg, then a bearish outside bar with a new high and a deep low.
    df = _bars(
        highs=[101, 105, 110, 116, 112, 111],
        lows=[99, 103, 108, 100, 104, 103],
        opens=[100, 104, 109, 111, 105, 105],
        closes=[100.5, 104.5, 109.5, 101, 110, 104],
    )
    piv = detect_pivots(df, PCT5, _atr_const(len(df)))
    assert (piv[1].idx, piv[1].kind.value, piv[1].price) == (3, "high", 116)
    assert (piv[2].idx, piv[2].kind.value, piv[2].price) == (3, "low", 100)


def test_outside_bar_bullish_order() -> None:
    # Bullish outside bar: low first (confirms previous high), then new high.
    df = _bars(
        highs=[101, 110, 109, 112, 111],
        lows=[99, 105, 103, 100, 108],
        opens=[100, 106, 108, 101, 110],
        closes=[100.5, 109, 104, 111, 109],
    )
    piv = detect_pivots(df, PCT5, _atr_const(len(df)))
    assert [(p.idx, p.kind.value) for p in piv][:3] == [(0, "low"), (1, "high"), (3, "low")]


def test_no_pivots_flat_market() -> None:
    df = _bars(highs=[100.2] * 30, lows=[99.8] * 30)
    assert detect_pivots(df, PCT5, _atr_const(30)) == []


def test_tick_order() -> None:
    df = _bars([2, 2], [1, 1], opens=[1.2, 1.8], closes=[1.8, 1.2])
    assert [t[2] for t in bar_ticks(df)] == [-1, 1, 1, -1]


def test_zigzag_ticks_direct() -> None:
    ticks = [(0, 10, -1), (1, 20, 1), (2, 15, -1), (3, 30, 1), (4, 5, -1)]
    raw, prov = zigzag_ticks(ticks, lambda b, p: 4.0)
    assert [r[1] for r in raw] == [10, 20, 15, 30, 5] and prov


def test_hierarchy_nested(settings) -> None:
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 800))
    c = c - c.min() + 50
    o = np.r_[c[0], c[:-1]]
    df = _bars(np.maximum(o, c) + rng.uniform(0, 0.5, 800), np.minimum(o, c) - rng.uniform(0, 0.5, 800), o, c)
    h = build_hierarchy(df, settings.pivots, settings.indicators)
    counts = [len(level) for level in h.levels]
    assert counts == sorted(counts) and counts[0] >= 2
    for coarse, fine in zip(h.levels[:-1], h.levels[1:]):
        fine_set = {(p.idx, p.kind) for p in fine}
        assert all((p.idx, p.kind) in fine_set for p in coarse)
    for level in h.levels:
        assert all(a.kind != b.kind for a, b in zip(level, level[1:]))
        assert level[-1].provisional
    assert h.inner(1, h.levels[0][0].idx, h.levels[0][1].idx) is not None


def test_hierarchy_percent_threshold(settings) -> None:
    s = with_overrides(settings, {"pivots": {"threshold": {"mode": "percent", "value": 5}}})
    df = _path([100, 130, 115, 150, 140, 160, 120])
    h = build_hierarchy(df, s.pivots, s.indicators)
    assert [round(p.price) for p in h.levels[-1]] == [100, 130, 115, 150, 140, 160, 120]
    assert [round(p.price) for p in h.levels[0]] == [100, 130, 115, 160, 120]  # 10 %: 150->140 entfällt


def test_plot_pivots(settings) -> None:
    df = _path([100, 130, 115, 150, 140, 160, 120])
    h = build_hierarchy(df, settings.pivots, settings.indicators)
    fig = plot_pivots(df, h, settings.plotting)
    assert len(fig.data) >= 2
