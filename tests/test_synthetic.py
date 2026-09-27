"""The synthetic generator produces the intended structure."""

from __future__ import annotations

import pytest

from elliott.data.loader import frame_to_market_data
from elliott.pivots.zigzag import build_hierarchy
from tests.synthetic import TEMPLATES, make_pattern, random_walk, sideways


@pytest.mark.parametrize("name", sorted(TEMPLATES))
@pytest.mark.parametrize("noise", [0.0, 0.01])
def test_top_pivots_detected(settings, name, noise) -> None:
    syn = make_pattern(name, bars=300, noise=noise, seed=3)
    frame_to_market_data(syn.df, settings.data)  # valid OHLC
    h = build_hierarchy(syn.df, settings.pivots, settings.indicators)
    top = {b for b, _ in syn.top_pivots}
    for level in h.levels:
        assert top <= {p.idx for p in level}


def test_template_ratios() -> None:
    syn = make_pattern("impulse", bars=300, height=47.64, depth=1)
    p = [price for _, price in syn.top_pivots]
    w1, w2, w3 = p[1] - p[0], p[1] - p[2], p[3] - p[2]
    assert w2 / w1 == pytest.approx(0.618, rel=1e-3)
    assert w3 / w1 == pytest.approx(1.618, rel=1e-3)


def test_down_direction_and_helpers(settings) -> None:
    syn = make_pattern("zigzag", direction=-1, bars=200)
    assert syn.top_pivots[-1][1] < syn.top_pivots[0][1]
    assert len(sideways(100)) == 100
    assert len(random_walk(500)) == 500
