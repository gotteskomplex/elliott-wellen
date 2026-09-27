"""Scoring: breakdown, penalties, softmax, textbook preference."""

from __future__ import annotations

import math

import pytest

from elliott.guidelines import GuideContext
from elliott.models import PatternType as P
from elliott.patterns import get_spec
from elliott.scoring import quick_score, score_view, softmax
from tests.helpers import IMPULSE_PTS, IMPULSE_TIMES, view


def test_breakdown_sums_to_guideline_score(settings) -> None:
    spec = get_spec(P.IMPULSE)
    sb = score_view(spec, view(P.IMPULSE, IMPULSE_PTS, IMPULSE_TIMES), GuideContext(), settings)
    applicable = [g for g in sb.guidelines if g.value is not None]
    assert sum(g.points for g in applicable) == pytest.approx(sb.guideline_score)
    assert 0 <= sb.total <= 100
    assert all(g.reason for g in sb.guidelines)
    names = {g.name for g in sb.guidelines}
    assert {"impulse_w3_ratio", "impulse_alternation", "proportionality", "completeness"} <= names


def test_quick_score_matches_full(settings) -> None:
    spec = get_spec(P.IMPULSE)
    v = view(P.IMPULSE, IMPULSE_PTS, IMPULSE_TIMES)
    assert quick_score(spec, v, GuideContext(), settings) == pytest.approx(score_view(spec, v, GuideContext(), settings).total)


def test_textbook_beats_distorted(settings) -> None:
    spec = get_spec(P.IMPULSE)
    good = score_view(spec, view(P.IMPULSE, IMPULSE_PTS, IMPULSE_TIMES), GuideContext(), settings).total
    distorted = [100.0, 120.0, 101.0, 125.0, 121.5, 160.0]
    bad = score_view(spec, view(P.IMPULSE, distorted, IMPULSE_TIMES), GuideContext(), settings).total
    assert good > bad + 10


def test_truncation_penalty(settings) -> None:
    spec = get_spec(P.IMPULSE)
    trunc = [100.0, 120.0, 107.64, 140.0, 127.64, 138.0]
    sb = score_view(spec, view(P.IMPULSE, trunc, IMPULSE_TIMES), GuideContext(), settings)
    assert any(p.name == "truncation" for p in sb.penalties)
    open_sb = score_view(spec, view(P.IMPULSE, trunc, IMPULSE_TIMES, last_open=True), GuideContext(), settings)
    assert not any(p.name == "truncation" for p in open_sb.penalties)


def test_rare_pattern_penalties(settings) -> None:
    run = score_view(get_spec(P.FLAT_RUNNING), view(P.FLAT_RUNNING, [100, 120, 95.28, 112]), GuideContext(), settings)
    assert any(p.name == "running_flat" for p in run.penalties)
    exp = score_view(get_spec(P.TRIANGLE_EXPANDING), view(P.TRIANGLE_EXPANDING, [100, 110, 104, 116, 98, 124]), GuideContext(), settings)
    assert any(p.name == "expanding_triangle" for p in exp.penalties)


def test_occam_complexity(settings) -> None:
    pts = [100.0, 120.0, 110.0, 130.0]
    zz = score_view(get_spec(P.ZIGZAG), view(P.ZIGZAG, pts), GuideContext(), settings)
    comb = score_view(get_spec(P.COMBINATION_WXY), view(P.COMBINATION_WXY, pts), GuideContext(), settings)
    assert zz.complexity < comb.complexity


def test_volume_neutral_without_data(settings) -> None:
    spec = get_spec(P.IMPULSE)
    sb = score_view(spec, view(P.IMPULSE, IMPULSE_PTS, IMPULSE_TIMES), GuideContext(volume=None), settings)
    vol = next(g for g in sb.guidelines if g.name == "impulse_volume")
    assert vol.value is None and "neutral" in vol.reason


def test_softmax(settings) -> None:
    w = softmax([90.0, 80.0, 70.0], settings.scoring.softmax_temperature)
    assert sum(w) == pytest.approx(1.0)
    assert w[0] > w[1] > w[2]
    assert softmax([], 5.0) == []
    hot = softmax([90.0, 80.0], 1000.0)
    assert math.isclose(hot[0], hot[1], rel_tol=0.02)


def test_context_guideline(settings) -> None:
    spec = get_spec(P.ZIGZAG)
    v = view(P.ZIGZAG, [100, 120, 110, 130])
    v.prior = 60.0
    ok = score_view(spec, v, GuideContext(), settings)
    v.prior = 5.0
    bad = score_view(spec, v, GuideContext(), settings)
    assert ok.total > bad.total
