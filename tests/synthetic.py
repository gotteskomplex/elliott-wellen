"""Generator for synthetic price series with known Elliott patterns.

A pattern is described by a *template*: a list of legs ``(price_move, time,
sub_pattern)`` in arbitrary units. The template is scaled to the requested
height and duration; every leg can recursively be subdivided into its own
textbook sub-pattern (motive legs into impulses, corrective legs into zigzags
or flats). The resulting waypoints are the ground truth.

Candles follow the piecewise linear path through the waypoints, optionally
with Gaussian noise and wicks. Waypoints are enforced as local extremes of
their adjacent legs, so the true pivots are known exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

Leg = tuple[float, float, str]

TEMPLATES: dict[str, list[Leg]] = {
    # Textbook impulse: W2 = 61,8 % (zigzag), W3 = 1,618 × W1, W4 = 38,2 % (flat), W5 = W1.
    "impulse": [(1.0, 1.0, "impulse"), (-0.618, 0.6, "zigzag"), (1.618, 1.5, "impulse"), (-0.618, 1.0, "flat"), (1.0, 1.0, "impulse")],
    "impulse_ext3": [(1.0, 1.0, "impulse"), (-0.618, 0.6, "zigzag"), (2.618, 2.2, "impulse"), (-1.0, 1.1, "flat"), (1.0, 1.0, "impulse")],
    "impulse_truncated": [(1.0, 1.0, "impulse"), (-0.618, 0.6, "zigzag"), (1.618, 1.5, "impulse"), (-0.618, 1.0, "flat"), (0.5, 0.6, "impulse")],
    "zigzag": [(1.0, 1.0, "impulse"), (-0.5, 0.6, "zigzag"), (1.0, 1.0, "impulse")],
    "flat_regular": [(1.0, 1.0, "zigzag"), (-0.95, 1.0, "zigzag"), (1.0, 1.0, "impulse")],
    "flat_expanded": [(1.0, 1.0, "zigzag"), (-1.236, 1.0, "zigzag"), (1.618, 1.2, "impulse")],
    "triangle": [(1.0, 1.0, "zigzag"), (-0.8, 0.9, "zigzag"), (0.618, 0.8, "zigzag"), (-0.5, 0.7, "zigzag"), (0.382, 0.6, "zigzag")],
    "ending_diagonal": [(1.0, 1.0, "zigzag"), (-0.66, 0.6, "zigzag"), (0.8, 0.8, "zigzag"), (-0.5, 0.5, "zigzag"), (0.6, 0.6, "zigzag")],
}


@dataclass
class SyntheticSeries:
    """Generated candles plus the ground-truth waypoints."""

    df: pd.DataFrame
    pattern: str
    top_pivots: list[tuple[int, float]]  # (bar, price) of the top-level pattern points P0..Pn
    all_waypoints: list[tuple[int, float]]


def _net(template: list[Leg]) -> float:
    return sum(leg[0] for leg in template)


def _build(kind: str, t0: float, p0: float, move: float, duration: float, depth: int) -> list[tuple[float, float, int]]:
    """Recursive waypoint construction -> list of (time, price, depth_level)."""
    if depth <= 0 or kind not in TEMPLATES:
        return [(t0 + duration, p0 + move, depth)]
    template = TEMPLATES[kind]
    scale_p = move / _net(template)
    scale_t = duration / sum(leg[1] for leg in template)
    out: list[tuple[float, float, int]] = []
    t, p = t0, p0
    for dp, dt, sub in template:
        leg_move, leg_dur = dp * scale_p, dt * scale_t
        out.extend(_build(sub, t, p, leg_move, leg_dur, depth - 1))
        t, p = t + leg_dur, p + leg_move
        out[-1] = (out[-1][0], out[-1][1], depth)  # leg end belongs to this degree
    return out


def make_pattern(
    pattern: str = "impulse",
    bars: int = 400,
    start_price: float = 100.0,
    height: float = 50.0,
    direction: int = 1,
    depth: int = 2,
    noise: float = 0.0,
    seed: int = 0,
    with_volume: bool = True,
    lead_in: int = 0,
    tail: int = 0,
    freq: str = "D",
) -> SyntheticSeries:
    """Create a synthetic series containing ``pattern``.

    Args:
        pattern: key of :data:`TEMPLATES`.
        bars: number of bars covered by the pattern.
        height: net price move of the pattern (before direction).
        direction: +1 up, -1 down.
        depth: 1 = only the top-level legs, 2 = one level of subwaves, ...
        noise: Gaussian noise std as fraction of ``height``.
        lead_in: bars of flat-ish data before the pattern starts.
        tail: bars after the pattern end (small drift against the last wave).
    """
    rng = np.random.default_rng(seed)
    net_move = direction * height
    raw = _build(pattern, 0.0, start_price, net_move, float(bars), depth)
    waypoints: list[tuple[float, float, int]] = [(0.0, start_price, depth)] + raw

    # map to integer bars, strictly increasing
    bar_idx: list[int] = []
    for t, _, _ in waypoints:
        b = int(round(t)) + lead_in
        if bar_idx and b <= bar_idx[-1]:
            b = bar_idx[-1] + 1
        bar_idx.append(b)
    prices = [p for _, p, _ in waypoints]
    levels = [lvl for _, _, lvl in waypoints]
    n_total = bar_idx[-1] + 1 + tail

    path = np.interp(np.arange(n_total), bar_idx, prices)
    if lead_in:
        path[:lead_in] = start_price
    if tail:
        last_leg = prices[-1] - prices[-2]
        path[bar_idx[-1] + 1 :] = prices[-1] - np.linspace(0, 0.1 * last_leg, tail + 1)[1:]
    sigma = noise * height
    close = path + (rng.normal(0.0, sigma, n_total) if sigma > 0 else 0.0)
    kinds = [0] * len(bar_idx)
    for i in range(len(bar_idx)):
        prev_p = prices[i - 1] if i > 0 else prices[i + 1]
        kinds[i] = 1 if prices[i] > prev_p else -1

    # Enforce waypoints as extremes of their adjacent legs.
    eps = max(1e-9, 1e-6 * abs(start_price))
    for i, (b, p) in enumerate(zip(bar_idx, prices)):
        lo_b = bar_idx[i - 1] if i > 0 else 0
        hi_b = bar_idx[i + 1] if i + 1 < len(bar_idx) else n_total - 1
        window = slice(lo_b, hi_b + 1)
        if kinds[i] > 0:
            close[window] = np.minimum(close[window], p - eps)
        else:
            close[window] = np.maximum(close[window], p + eps)
    for b, p in zip(bar_idx, prices):
        close[b] = p

    open_ = np.r_[close[0], close[:-1]]
    body_hi, body_lo = np.maximum(open_, close), np.minimum(open_, close)
    wick = np.abs(rng.normal(0.0, sigma * 0.5, n_total)) if sigma > 0 else np.zeros(n_total)
    high = body_hi + wick
    low = body_lo - wick
    for i, (b, p) in enumerate(zip(bar_idx, prices)):
        lo_b = bar_idx[i - 1] if i > 0 else 0
        hi_b = bar_idx[i + 1] if i + 1 < len(bar_idx) else n_total - 1
        window = np.arange(lo_b, hi_b + 1)
        window = window[window != b]
        if kinds[i] > 0:
            high[window] = np.minimum(high[window], p - eps)
            high[b] = p
        else:
            low[window] = np.maximum(low[window], p + eps)
            low[b] = p
    high = np.maximum(high, body_hi)
    low = np.minimum(low, body_lo)
    if np.any(low <= 0):
        raise ValueError("Synthetische Reihe wird negativ – start_price erhöhen")

    df = pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close},
        index=pd.date_range("2020-01-01", periods=n_total, freq=freq),
    )
    if with_volume:
        step = np.abs(np.diff(path, prepend=path[0]))
        df["volume"] = 1000.0 + 50000.0 * step / max(height, 1e-9) + rng.uniform(0, 50, n_total)
    top = [(b, p) for b, p, lvl in zip(bar_idx, prices, levels) if lvl == depth]
    return SyntheticSeries(df=df, pattern=pattern, top_pivots=top, all_waypoints=list(zip(bar_idx, prices)))


def sideways(bars: int = 300, price: float = 100.0, noise: float = 0.01, seed: int = 0) -> pd.DataFrame:
    """Trendless noise around a level (mean-reverting)."""
    rng = np.random.default_rng(seed)
    x = np.zeros(bars)
    for i in range(1, bars):
        x[i] = 0.9 * x[i - 1] + rng.normal(0.0, noise)
    close = price * (1.0 + x)
    open_ = np.r_[close[0], close[:-1]]
    wick = np.abs(rng.normal(0.0, noise * price * 0.3, bars))
    return pd.DataFrame(
        {"open": open_, "high": np.maximum(open_, close) + wick, "low": np.minimum(open_, close) - wick, "close": close},
        index=pd.date_range("2021-01-01", periods=bars, freq="D"),
    )


def random_walk(bars: int = 2000, price: float = 100.0, vol: float = 0.015, seed: int = 0) -> pd.DataFrame:
    """Geometric random walk (used for benchmarks)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0002, vol, bars)
    close = price * np.exp(np.cumsum(rets))
    open_ = np.r_[price, close[:-1]]
    wick = np.abs(rng.normal(0.0, vol * 0.5, bars)) * close
    return pd.DataFrame(
        {
            "open": open_, "high": np.maximum(open_, close) + wick, "low": np.minimum(open_, close) - wick,
            "close": close, "volume": rng.uniform(1e5, 2e5, bars),
        },
        index=pd.date_range("2018-01-01", periods=bars, freq="D"),
    )
