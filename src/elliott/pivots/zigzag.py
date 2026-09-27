"""ZigZag turning-point detection on High/Low data with several degrees.

Algorithm
---------
Every bar is split into two *ticks* (its low and its high). The order of the
ticks inside a bar is estimated from the candle body: a rising candle
(``close >= open``) is assumed to print its low first, a falling candle its
high first. This handles outside bars (new high *and* new low in one bar)
consistently: both extremes are processed in a plausible order and two pivots
on the same bar are possible.

The classic ZigZag state machine then runs over the ticks: in an up-leg the
candidate high is raised until price falls by at least the threshold from it,
which confirms the high as a pivot and starts a down-leg (and vice versa).
Equal highs/lows keep the *first* occurrence (strict comparison).

The last candidate at the right edge is not confirmed; it is returned as a
``provisional`` pivot.

Degrees
-------
The finest degree is computed from the bars. Every coarser degree is computed
by running the same state machine over the pivots of the next finer degree
with a larger threshold. This guarantees a strict hierarchy: every coarse
pivot is also a pivot of all finer degrees, which the subwave validation
relies on.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from elliott.data.indicators import atr
from elliott.models import Pivot, PivotKind
from elliott.settings import IndicatorConfig, PivotConfig, ThresholdSpec

# (bar index, price, kind sign +1 high / -1 low)
Tick = tuple[int, float, int]
RawPivot = tuple[int, float, int]
ThresholdFn = Callable[[int, float], float]


def bar_ticks(df: pd.DataFrame) -> list[Tick]:
    """Split bars into ordered high/low ticks (see module docstring)."""
    opens = df["open"].to_numpy(dtype=float)
    highs = df["high"].to_numpy(dtype=float)
    lows = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    ticks: list[Tick] = []
    for i in range(len(df)):
        if closes[i] >= opens[i]:
            ticks.append((i, lows[i], -1))
            ticks.append((i, highs[i], 1))
        else:
            ticks.append((i, highs[i], 1))
            ticks.append((i, lows[i], -1))
    return ticks


def zigzag_ticks(ticks: Iterable[Tick], threshold: ThresholdFn) -> tuple[list[RawPivot], bool]:
    """Run the ZigZag state machine.

    Args:
        ticks: ordered ticks ``(bar, price, kind)``.
        threshold: function ``(bar, price) -> absolute reversal distance`` for
            an extreme at ``bar`` with price ``price``.

    Returns:
        ``(pivots, has_provisional)``; the last pivot is provisional if
        ``has_provisional`` is true. Pivots strictly alternate high/low.
    """
    pivots: list[RawPivot] = []
    trend = 0
    hi: tuple[int, float] | None = None
    lo: tuple[int, float] | None = None
    hi_seq = lo_seq = -1
    cand: tuple[int, float] | None = None

    for seq, (bar, price, kind) in enumerate(ticks):
        if trend == 0:
            if hi is None or lo is None:
                hi = lo = (bar, price)
                hi_seq = lo_seq = seq
                continue
            if kind > 0 and price > hi[1]:
                hi, hi_seq = (bar, price), seq
            if kind < 0 and price < lo[1]:
                lo, lo_seq = (bar, price), seq
            if lo_seq < hi_seq and hi[1] - lo[1] >= threshold(*lo):
                pivots.append((lo[0], lo[1], -1))
                trend, cand = 1, hi
            elif hi_seq < lo_seq and hi[1] - lo[1] >= threshold(*hi):
                pivots.append((hi[0], hi[1], 1))
                trend, cand = -1, lo
            continue

        assert cand is not None
        if trend > 0:
            if kind > 0 and price > cand[1]:
                cand = (bar, price)
            elif kind < 0 and cand[1] - price >= threshold(*cand):
                pivots.append((cand[0], cand[1], 1))
                trend, cand = -1, (bar, price)
        else:
            if kind < 0 and price < cand[1]:
                cand = (bar, price)
            elif kind > 0 and price - cand[1] >= threshold(*cand):
                pivots.append((cand[0], cand[1], -1))
                trend, cand = 1, (bar, price)

    if trend == 0 or cand is None:
        return [], False
    pivots.append((cand[0], cand[1], trend))
    return pivots, True


def make_threshold_fn(spec: ThresholdSpec, atr_values: np.ndarray) -> ThresholdFn:
    """Build the reversal-distance function for a threshold spec."""
    if spec.mode == "percent":
        frac = spec.value / 100.0
        return lambda bar, price: price * frac
    mult = spec.value
    return lambda bar, price: mult * float(atr_values[bar])


@dataclass
class PivotHierarchy:
    """Pivots of several degrees, ordered from coarse (0) to fine (-1)."""

    levels: list[list[Pivot]]
    thresholds: list[ThresholdSpec]
    names: list[str]

    @property
    def finest(self) -> int:
        return len(self.levels) - 1

    def inner(self, level: int, start_idx: int, end_idx: int) -> list[Pivot]:
        """Pivots of ``level`` strictly between two bar indices."""
        return [p for p in self.levels[level] if start_idx < p.idx < end_idx]


def _to_models(raw: Sequence[RawPivot], provisional: bool, index: pd.DatetimeIndex, level: int) -> list[Pivot]:
    out: list[Pivot] = []
    last = len(raw) - 1
    for i, (bar, price, kind) in enumerate(raw):
        out.append(
            Pivot(
                idx=int(bar),
                ts=index[bar].to_pydatetime(),
                price=float(price),
                kind=PivotKind.from_sign(kind),
                level=level,
                provisional=provisional and i == last,
            )
        )
    return out


def detect_pivots(df: pd.DataFrame, spec: ThresholdSpec, atr_values: np.ndarray, level: int = 0) -> list[Pivot]:
    """Single-degree ZigZag on bar data."""
    raw, prov = zigzag_ticks(bar_ticks(df), make_threshold_fn(spec, atr_values))
    return _to_models(raw, prov, df.index, level)


def build_hierarchy(df: pd.DataFrame, pivot_cfg: PivotConfig, ind_cfg: IndicatorConfig) -> PivotHierarchy:
    """Compute pivots for all configured degrees (strictly nested)."""
    atr_values = atr(df, ind_cfg.atr_period).to_numpy(dtype=float)
    specs = [pivot_cfg.threshold.scaled(m) for m in pivot_cfg.degree_multipliers]
    names = list(pivot_cfg.degree_names)
    while len(names) < len(specs):
        names.append(f"Grad {len(names)}")
    n_levels = len(specs)

    raw_levels: list[tuple[list[RawPivot], bool]] = [([], False)] * n_levels
    finest = n_levels - 1
    raw_levels[finest] = zigzag_ticks(bar_ticks(df), make_threshold_fn(specs[finest], atr_values))
    for lvl in range(finest - 1, -1, -1):
        finer, _ = raw_levels[lvl + 1]
        raw_levels[lvl] = zigzag_ticks(finer, make_threshold_fn(specs[lvl], atr_values))

    levels = [_to_models(raw, prov, df.index, lvl) for lvl, (raw, prov) in enumerate(raw_levels)]
    return PivotHierarchy(levels=levels, thresholds=specs, names=names[:n_levels])
