"""German number formatting helpers for reasons and reports."""

from __future__ import annotations

import math


def _german(text: str) -> str:
    return text.replace(",", " ").replace(".", ",").replace(" ", ".")


def fmt_price(price: float) -> str:
    """Format a price with sensible precision and German separators (``41.230,50``)."""
    if not math.isfinite(price):
        return "–"
    a = abs(price)
    if a >= 1000:
        text = f"{price:,.2f}"
    elif a >= 1:
        text = f"{price:,.2f}" if a >= 100 else f"{price:,.3f}"
    elif a == 0:
        text = "0"
    else:
        digits = max(2, 3 - int(math.floor(math.log10(a))))
        text = f"{price:.{digits}f}"
    return _german(text)


def fmt_pct(r: float) -> str:
    """Ratio -> German percent string (``0.618 -> '61,8 %'``)."""
    if not math.isfinite(r):
        return "∞"
    return _german(f"{r * 100:.1f}") + " %"


def fmt_ratio(r: float) -> str:
    """Ratio -> German decimal string (``1.618 -> '1,618'``)."""
    if not math.isfinite(r):
        return "∞"
    return _german(f"{r:.3f}")
