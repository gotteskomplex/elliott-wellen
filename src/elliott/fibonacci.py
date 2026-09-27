"""Fibonacci retracements/extensions and tolerance-band scoring.

All functions accept a :class:`~elliott.measure.PriceScale`, so the same code
works in linear and logarithmic space.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from elliott.fmt import fmt_pct, fmt_ratio  # noqa: F401  (re-export)
from elliott.measure import PriceScale
from elliott.settings import FibConfig


def length(a: float, b: float, scale: PriceScale) -> float:
    """Absolute length of the move ``a -> b`` in measurement space."""
    return abs(scale.tr(b) - scale.tr(a))


def ratio(numerator: float, denominator: float) -> float:
    """Safe ratio (``inf`` if the denominator is zero)."""
    if denominator == 0:
        return math.inf
    return numerator / denominator


def retracement_level(start: float, end: float, r: float, scale: PriceScale) -> float:
    """Price that retraces ``r`` of the move ``start -> end``."""
    s, e = scale.tr(start), scale.tr(end)
    return scale.inv(e - r * (e - s))


def retracement_ratio(start: float, end: float, retrace_end: float, scale: PriceScale) -> float:
    """How much of ``start -> end`` has been retraced at ``retrace_end``."""
    s, e, r = scale.tr(start), scale.tr(end), scale.tr(retrace_end)
    return ratio(e - r, e - s)


def extension_level(base_start: float, base_end: float, origin: float, r: float, scale: PriceScale) -> float:
    """Project ``r`` times the move ``base_start -> base_end`` from ``origin``."""
    b0, b1, o = scale.tr(base_start), scale.tr(base_end), scale.tr(origin)
    return scale.inv(o + r * (b1 - b0))


def band_score(value: float, target: float, cfg: FibConfig) -> float:
    """Continuous closeness score in [0, 1].

    Full score within ``±tolerance`` (relative) of ``target``; outside the band
    the score decays like a Gaussian with width ``decay_sigma``.
    """
    if target == 0 or not math.isfinite(value):
        return 0.0
    rel = abs(value / target - 1.0)
    if rel <= cfg.tolerance:
        return 1.0
    z = (rel - cfg.tolerance) / cfg.decay_sigma
    return math.exp(-0.5 * z * z)


def fib_score(value: float, targets: Sequence[float], cfg: FibConfig) -> tuple[float, float]:
    """Best band score over several target ratios -> ``(score, best_target)``."""
    best, best_t = 0.0, targets[0] if targets else math.nan
    for t in targets:
        s = band_score(value, t, cfg)
        if s > best:
            best, best_t = s, t
    return best, best_t


def range_score(value: float, lo: float, hi: float, cfg: FibConfig) -> float:
    """1 inside ``[lo, hi]`` (widened by the tolerance), Gaussian decay outside."""
    if not math.isfinite(value):
        return 0.0
    lo_t, hi_t = lo * (1 - cfg.tolerance), hi * (1 + cfg.tolerance)
    if lo_t <= value <= hi_t:
        return 1.0
    edge = lo if value < lo_t else hi
    rel = abs(value / edge - 1.0) - cfg.tolerance
    z = rel / cfg.decay_sigma
    return math.exp(-0.5 * z * z)
