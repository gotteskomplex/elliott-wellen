"""Projection: targets, confluence, invalidation levels, alternatives."""

from __future__ import annotations

import pytest

from elliott.data.loader import frame_to_market_data
from elliott.models import PatternType as P
from elliott.patterns import get_spec
from elliott.pipeline import analyze
from elliott.projection import _Target, cluster_targets, invalidations, wave_targets
from elliott.measure import PriceScale
from tests.helpers import IMPULSE_PTS, IMPULSE_TIMES, view
from tests.synthetic import make_pattern


def _result(settings, pattern="impulse", cut=None, **kw):
    syn = make_pattern(pattern, bars=300, **kw)
    df = syn.df if cut is None else syn.df.iloc[: cut + 1]
    return syn, analyze(frame_to_market_data(df, settings.data), settings)


def _find(result, pattern: P, truth_bars: list[int], allow_running_next: bool = False):
    """Scenario whose points equal ``truth_bars`` – optionally followed by one
    more, still running wave (the next wave has just started)."""
    for s in result.scenarios:
        bars = [s.count.waves[0].start.idx] + [w.end.idx for w in s.count.waves]
        if s.count.pattern is not pattern:
            continue
        if bars == truth_bars:
            return s
        if allow_running_next and bars[:-1] == truth_bars and s.projection.current_wave_running:
            return s
    return None


def test_wave2_invalidation_is_w1_start(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    w2_end = syn.top_pivots[2][0]
    _, res = _result(settings, cut=w2_end + 2)
    truth = [b for b, _ in syn.top_pivots[:3]]
    s = _find(res, P.IMPULSE, truth, allow_running_next=True)
    assert s is not None, [(x.count.pattern, len(x.count.waves)) for x in res.scenarios[:8]]
    inv = s.projection.invalidation
    assert inv.hard and inv.rule == "w2_max_retrace"
    assert inv.price == pytest.approx(syn.top_pivots[0][1], rel=1e-6)
    assert inv.side == "below" and "Unter" in inv.reason and "Start von Welle 1" in inv.reason
    # next wave 3 targets: 1.618 × W1 from the end of W2
    p0, p1, p2 = (p for _, p in syn.top_pivots[:3])
    expected = p2 + 1.618 * (p1 - p0)
    assert any(z.low <= expected <= z.high for z in s.projection.targets)
    assert s.projection.next_direction.value == "up"


def test_wave4_invalidation_is_overlap(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    w3_end = syn.top_pivots[3][0]
    _, res = _result(settings, cut=w3_end + 2)
    truth = [b for b, _ in syn.top_pivots[:4]]
    s = _find(res, P.IMPULSE, truth, allow_running_next=True)
    assert s is not None
    hard = [i for i in s.projection.invalidations if i.hard]
    assert any(i.rule == "w4_no_overlap" and i.price == pytest.approx(syn.top_pivots[1][1], rel=1e-6) for i in hard)


def test_every_scenario_is_complete(settings) -> None:
    _, res = _result(settings, noise=0.01, seed=4)
    assert res.scenarios
    for s in res.scenarios:
        p = s.projection
        assert p is not None and p.position and p.next_move
        assert p.invalidation.reason and p.invalidation.price > 0
        assert s.score.guidelines and 0 <= s.score.total <= 100
        assert p.targets, s.id
        assert p.alternative


def test_probabilities_top_n(settings) -> None:
    _, res = _result(settings)
    probs = [s.probability for s in res.scenarios if s.probability is not None]
    assert len(probs) == min(settings.scoring.top_n, len(res.scenarios))
    assert sum(probs) == pytest.approx(1.0)


def test_cluster_confluence(settings) -> None:
    targets = [_Target(110.0, "a", 0), _Target(110.5, "b", 1), _Target(120.0, "c", 0), _Target(95.0, "d", 0)]
    zones = cluster_targets(targets, current=100.0, direction=1, cfg=settings)
    assert zones[0].confluence == 2 and zones[0].highlighted
    assert all(z.center > 100 for z in zones)  # targets behind the price are dropped


def test_wave_targets_impulse(settings) -> None:
    spec = get_spec(P.IMPULSE)
    v = view(P.IMPULSE, IMPULSE_PTS[:3], IMPULSE_TIMES[:3])
    t = wave_targets(spec, v, 3, None, settings)
    assert any(abs(x - (107.64 + 1.618 * 20)) < 1e-6 for x, _, _ in t)


def test_invalidation_down_trend_log(settings) -> None:
    spec = get_spec(P.IMPULSE)
    scale = PriceScale(True)
    import math

    pts = [200.0, 150.0, 180.0]
    v = view(P.IMPULSE, [math.log(p) for p in pts], [0, 10, 20], direction=-1)
    v = v.with_points(v.x, v.t, tuple(pts), last_open=True)
    levels = invalidations(spec, v, running=False, now_bar=21, scale=scale, cfg=settings)
    hard = [lv for lv in levels if lv.hard]
    assert hard and hard[0].side == "above" and hard[0].price == pytest.approx(200.0, rel=1e-6)


def test_running_wave_has_structural_and_hard_levels(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    a, b = syn.top_pivots[4][0], syn.top_pivots[5][0]
    _, res = _result(settings, cut=a + int(0.75 * (b - a)))
    s = next(x for x in res.scenarios if x.count.pattern is P.IMPULSE and len(x.count.waves) == 5 and x.projection.current_wave_running)
    kinds = {i.hard for i in s.projection.invalidations}
    assert kinds == {True, False}
    assert "läuft noch" in s.projection.position


def test_alternative_links_to_surviving_scenario(settings) -> None:
    _, res = _result(settings, cut=250)
    ids = {s.id for s in res.scenarios}
    for s in res.scenarios:
        if s.projection.alternative_scenario_id:
            assert s.projection.alternative_scenario_id in ids
            assert s.projection.alternative_scenario_id != s.id
