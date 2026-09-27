"""Test helpers."""

from __future__ import annotations

from collections.abc import Sequence

from elliott.measure import CountView, make_view
from elliott.models import PATTERN_LABELS, PatternType


def view(
    pattern: PatternType,
    points: Sequence[float],
    times: Sequence[int] | None = None,
    last_open: bool = False,
    subs: tuple[PatternType | None, ...] | None = None,
    direction: int = 1,
) -> CountView:
    """Build a CountView from *price* points (linear scale)."""
    n = len(PATTERN_LABELS[pattern])
    times = list(times) if times is not None else [10 * i for i in range(len(points))]
    return make_view(pattern, n, direction, list(points), times, list(points), last_open, subs)


# Normalized textbook point sets (P0..Pn), linear prices
IMPULSE_PTS = [100.0, 120.0, 107.64, 140.0, 127.64, 147.64]
IMPULSE_TIMES = [0, 20, 32, 62, 82, 102]
ZIGZAG_PTS = [100.0, 120.0, 110.0, 130.0]
FLAT_REG_PTS = [100.0, 120.0, 101.0, 121.0]
FLAT_EXP_PTS = [100.0, 120.0, 95.28, 127.64]
FLAT_RUN_PTS = [100.0, 120.0, 95.28, 112.0]
TRIANGLE_PTS = [100.0, 120.0, 104.0, 116.36, 106.36, 114.0]
TRIANGLE_TIMES = [0, 20, 38, 54, 68, 80]
DIAGONAL_PTS = [100.0, 120.0, 106.8, 122.8, 112.8, 124.8]
DIAGONAL_TIMES = [0, 20, 32, 48, 58, 70]
