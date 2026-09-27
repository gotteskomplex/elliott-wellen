"""Price measurement (linear/log) and the lightweight count view used by rules.

All lengths, retracements and Fibonacci ratios are computed in a *measurement
space*: raw prices (linear) or natural-log prices (log scale). Order
comparisons (e.g. "wave 4 must not enter wave 1") are invariant under this
monotone transform, ratios are not – which is exactly why long-term charts and
crypto need the log option.

:class:`CountView` normalizes a count so that its first wave always points
"up" (``x = direction * transformed_price``). Rules and guidelines can then be
written once for both directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from elliott.models import PatternType


@dataclass(frozen=True, slots=True)
class PriceScale:
    """Linear or logarithmic price measurement."""

    log: bool

    def tr(self, price: float) -> float:
        """Transform a price into measurement space."""
        return math.log(price) if self.log else float(price)

    def tr_array(self, prices: np.ndarray) -> np.ndarray:
        return np.log(prices) if self.log else np.asarray(prices, dtype=float)

    def inv(self, value: float) -> float:
        """Transform back to a price."""
        return math.exp(value) if self.log else float(value)

    @property
    def name(self) -> str:
        return "log" if self.log else "linear"


@dataclass(slots=True)
class CountView:
    """Normalized, direction-agnostic view of a (partial) count.

    Attributes:
        pattern: pattern type being counted.
        d: direction of the first wave (+1 up, -1 down) in price terms.
        x: normalized transformed prices of the points P0..Pk
            (``x_i = d * tr(price_i)``, so wave 1 always rises in ``x``).
        t: bar indices of the points.
        raw: raw prices of the points (for human readable reasons).
        n: number of waves of the complete pattern.
        last_open: the last wave ends at the provisional right-edge pivot and
            may still extend; "must reach" rules for it are deferred.
        sub_patterns: known pattern types of the subwaves (``None`` = unknown).
    """

    pattern: PatternType
    d: int
    x: tuple[float, ...]
    t: tuple[int, ...]
    raw: tuple[float, ...]
    n: int
    last_open: bool = False
    sub_patterns: tuple[PatternType | None, ...] | None = None

    @property
    def k(self) -> int:
        """Number of waves present."""
        return len(self.x) - 1

    @property
    def complete(self) -> bool:
        return self.k == self.n

    def has(self, wave: int) -> bool:
        """True if wave ``wave`` (1-based) is present."""
        return 1 <= wave <= self.k

    def L(self, wave: int) -> float:
        """Length of wave ``wave`` (1-based) in measurement space."""
        return abs(self.x[wave] - self.x[wave - 1])

    def dur(self, wave: int) -> int:
        """Duration of a wave in bars (at least 1)."""
        return max(1, self.t[wave] - self.t[wave - 1])

    def is_open(self, wave: int) -> bool:
        """True if ``wave`` is the last wave and may still extend."""
        return self.last_open and wave == self.k

    def wave_up(self, wave: int) -> bool:
        """Direction of a wave in normalized space (odd waves rise)."""
        return wave % 2 == 1

    def price(self, point: int) -> float:
        return self.raw[point]

    def with_points(self, x: tuple[float, ...], t: tuple[int, ...], raw: tuple[float, ...], last_open: bool) -> "CountView":
        """Copy with replaced points (used for hypothetical extensions)."""
        subs = self.sub_patterns
        if subs is not None:
            subs = tuple(subs[: len(x) - 1]) + (None,) * max(0, len(x) - 1 - len(subs))
        return CountView(self.pattern, self.d, x, t, raw, self.n, last_open, subs)


def make_view(
    pattern: PatternType,
    n: int,
    d: int,
    tp: list[float] | tuple[float, ...],
    bars: list[int] | tuple[int, ...],
    raw: list[float] | tuple[float, ...],
    last_open: bool = False,
    sub_patterns: tuple[PatternType | None, ...] | None = None,
) -> CountView:
    """Build a :class:`CountView` from transformed prices ``tp``."""
    return CountView(
        pattern=pattern,
        d=d,
        x=tuple(d * v for v in tp),
        t=tuple(int(b) for b in bars),
        raw=tuple(float(r) for r in raw),
        n=n,
        last_open=last_open,
        sub_patterns=sub_patterns,
    )
