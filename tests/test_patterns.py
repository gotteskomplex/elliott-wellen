"""Pattern specs: textbook point sets pass exactly the expected patterns."""

from __future__ import annotations

import pytest

from elliott.models import PatternType as P
from elliott.patterns import SPECS, get_spec
from elliott.rules import passes
from tests.helpers import (
    DIAGONAL_PTS, DIAGONAL_TIMES, FLAT_EXP_PTS, FLAT_REG_PTS, FLAT_RUN_PTS, IMPULSE_PTS, IMPULSE_TIMES,
    TRIANGLE_PTS, TRIANGLE_TIMES, ZIGZAG_PTS, view,
)


def matching(points, times, settings, n_waves: int) -> set[P]:
    out = set()
    for spec in SPECS.values():
        if spec.n != n_waves:
            continue
        if passes(spec.all_rules, view(spec.type, points, times), settings.rules):
            out.add(spec.type)
    return out


def test_impulse_points(settings) -> None:
    m = matching(IMPULSE_PTS, IMPULSE_TIMES, settings, 5)
    assert P.IMPULSE in m
    assert P.ENDING_DIAGONAL not in m and P.TRIANGLE_CONTRACTING not in m


def test_diagonal_points(settings) -> None:
    m = matching(DIAGONAL_PTS, DIAGONAL_TIMES, settings, 5)
    assert {P.ENDING_DIAGONAL, P.LEADING_DIAGONAL} <= m
    assert P.IMPULSE not in m


def test_triangle_points(settings) -> None:
    m = matching(TRIANGLE_PTS, TRIANGLE_TIMES, settings, 5)
    assert P.TRIANGLE_CONTRACTING in m
    assert P.IMPULSE not in m and P.TRIANGLE_EXPANDING not in m


@pytest.mark.parametrize(
    ("points", "expected", "excluded"),
    [
        (ZIGZAG_PTS, P.ZIGZAG, {P.FLAT_REGULAR, P.FLAT_EXPANDED}),
        (FLAT_REG_PTS, P.FLAT_REGULAR, {P.FLAT_EXPANDED, P.FLAT_RUNNING}),
        (FLAT_EXP_PTS, P.FLAT_EXPANDED, {P.ZIGZAG, P.FLAT_REGULAR, P.FLAT_RUNNING}),
        (FLAT_RUN_PTS, P.FLAT_RUNNING, {P.ZIGZAG, P.FLAT_EXPANDED}),
    ],
)
def test_three_wave_points(settings, points, expected, excluded) -> None:
    m = matching(points, None, settings, 3)
    assert expected in m
    assert not (m & excluded)


def test_spec_consistency() -> None:
    for spec in SPECS.values():
        assert len(spec.labels) == spec.n == len(spec.sub_types)
    assert get_spec(P.IMPULSE).is_motive and not get_spec(P.ZIGZAG).is_motive
    # triangles never as wave 2 of an impulse
    assert not (get_spec(P.IMPULSE).sub_types[1] & {P.TRIANGLE_CONTRACTING, P.TRIANGLE_BARRIER})
