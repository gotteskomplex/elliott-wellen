from __future__ import annotations

import math

import pytest

from elliott.fibonacci import (
    band_score, extension_level, fib_score, length, range_score, retracement_level, retracement_ratio,
)
from elliott.fmt import fmt_pct, fmt_price, fmt_ratio
from elliott.measure import PriceScale

LIN, LOG = PriceScale(False), PriceScale(True)


def test_retracement_linear() -> None:
    assert retracement_level(100, 200, 0.618, LIN) == pytest.approx(138.2)
    assert retracement_ratio(100, 200, 150, LIN) == pytest.approx(0.5)


def test_retracement_log() -> None:
    # 50 % in log space of 100 -> 400 is the geometric mean 200.
    assert retracement_level(100, 400, 0.5, LOG) == pytest.approx(200)
    assert retracement_ratio(100, 400, 200, LOG) == pytest.approx(0.5)


def test_extension_linear_and_log() -> None:
    assert extension_level(100, 120, 110, 1.618, LIN) == pytest.approx(142.36)
    # log: 1.0 × (100 -> 200) from 150 -> 300
    assert extension_level(100, 200, 150, 1.0, LOG) == pytest.approx(300)
    assert length(100, 200, LOG) == pytest.approx(math.log(2))


def test_down_moves() -> None:
    assert retracement_level(200, 100, 0.5, LIN) == pytest.approx(150)
    assert extension_level(200, 150, 180, 1.0, LIN) == pytest.approx(130)


def test_band_score(settings) -> None:
    cfg = settings.fib
    assert band_score(1.618, 1.618, cfg) == 1.0
    assert band_score(1.618 * (1 + cfg.tolerance * 0.9), 1.618, cfg) == 1.0
    s1 = band_score(1.618 * 1.08, 1.618, cfg)
    s2 = band_score(1.618 * 1.2, 1.618, cfg)
    assert 0 < s2 < s1 < 1  # continuous decay


def test_fib_and_range_score(settings) -> None:
    cfg = settings.fib
    s, t = fib_score(2.6, [1.618, 2.618, 1.0], cfg)
    assert t == 2.618 and s == 1.0
    assert range_score(0.55, 0.5, 0.618, cfg) == 1.0
    assert 0 < range_score(0.75, 0.5, 0.618, cfg) < 1
    assert range_score(float("nan"), 0.5, 0.6, cfg) == 0.0


def test_formatting() -> None:
    assert fmt_price(41230.5) == "41.230,50"
    assert fmt_pct(0.618) == "61,8 %"
    assert fmt_ratio(1.618) == "1,618"
    assert fmt_price(0.000123) .startswith("0,000123")
