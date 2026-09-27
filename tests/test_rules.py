"""Every hard rule with a positive and a negative example."""

from __future__ import annotations

import pytest

from elliott import rules as R
from elliott.models import PatternType as P
from tests.helpers import (
    DIAGONAL_PTS, DIAGONAL_TIMES, FLAT_EXP_PTS, FLAT_REG_PTS, FLAT_RUN_PTS, IMPULSE_PTS, IMPULSE_TIMES,
    TRIANGLE_PTS, TRIANGLE_TIMES, ZIGZAG_PTS, view,
)


@pytest.fixture
def rc(settings):
    return settings.rules


def _check(rule, v, rc, expected: bool) -> None:
    passed, reason = rule(v, rc)  # rules unpack like (bool, str)
    assert passed is expected, reason
    assert isinstance(reason, str) and reason


# --- common ---------------------------------------------------------------

def test_alternation(rc) -> None:
    _check(R.rule_alternation, view(P.IMPULSE, IMPULSE_PTS), rc, True)
    _check(R.rule_alternation, view(P.IMPULSE, [100, 120, 125]), rc, False)


def test_subwave_types(rc) -> None:
    ok = view(P.IMPULSE, IMPULSE_PTS, subs=(P.IMPULSE, P.ZIGZAG, P.IMPULSE, P.TRIANGLE_CONTRACTING, P.ENDING_DIAGONAL))
    _check(R.rule_subwave_types, ok, rc, True)
    tri_w2 = view(P.IMPULSE, IMPULSE_PTS, subs=(None, P.TRIANGLE_CONTRACTING, None, None, None))
    _check(R.rule_subwave_types, tri_w2, rc, False)
    ending_w1 = view(P.IMPULSE, IMPULSE_PTS, subs=(P.ENDING_DIAGONAL, None, None, None, None))
    _check(R.rule_subwave_types, ending_w1, rc, False)
    leading_c = view(P.ZIGZAG, ZIGZAG_PTS, subs=(None, None, P.LEADING_DIAGONAL))
    _check(R.rule_subwave_types, leading_c, rc, False)


# --- impulse ----------------------------------------------------------------

def test_w2_max_retrace(rc) -> None:
    _check(R.rule_w2_max_retrace, view(P.IMPULSE, [100, 120, 101]), rc, True)
    _check(R.rule_w2_max_retrace, view(P.IMPULSE, [100, 120, 99]), rc, False)


def test_w2_max_retrace_downtrend(rc) -> None:
    v = view(P.IMPULSE, [200, 180, 201], direction=-1)
    passed, reason = R.rule_w2_max_retrace(v, rc)
    assert not passed and "überschreitet" in reason


def test_w3_beyond_w1(rc) -> None:
    _check(R.rule_w3_beyond_w1, view(P.IMPULSE, [100, 120, 110, 125]), rc, True)
    _check(R.rule_w3_beyond_w1, view(P.IMPULSE, [100, 120, 110, 118]), rc, False)
    res = R.rule_w3_beyond_w1(view(P.IMPULSE, [100, 120, 110, 118], last_open=True), rc)
    assert res.passed and res.deferred


def test_w3_not_shortest(rc) -> None:
    _check(R.rule_w3_not_shortest, view(P.IMPULSE, IMPULSE_PTS), rc, True)
    # W1 = 20, W3 = 15, W5 = 18 -> W3 shortest
    _check(R.rule_w3_not_shortest, view(P.IMPULSE, [100, 120, 110, 125, 121, 139]), rc, False)
    # open wave 5 already too long -> fails immediately (limit)
    _check(R.rule_w3_not_shortest, view(P.IMPULSE, [100, 120, 110, 125, 121, 139], last_open=True), rc, False)


def test_w4_no_overlap(rc) -> None:
    _check(R.rule_w4_no_overlap, view(P.IMPULSE, [100, 120, 110, 140, 121]), rc, True)
    _check(R.rule_w4_no_overlap, view(P.IMPULSE, [100, 120, 110, 140, 119]), rc, False)
    _check(R.rule_w4_no_overlap, view(P.IMPULSE, [100, 120, 110, 140, 119], last_open=True), rc, False)


# --- diagonal ---------------------------------------------------------------

def test_diagonal_w4_max_retrace(rc) -> None:
    _check(R.rule_diagonal_w4_max_retrace, view(P.ENDING_DIAGONAL, DIAGONAL_PTS, DIAGONAL_TIMES), rc, True)
    _check(R.rule_diagonal_w4_max_retrace, view(P.ENDING_DIAGONAL, [100, 120, 106, 122, 105]), rc, False)


def test_diagonal_w4_overlap(rc) -> None:
    _check(R.rule_diagonal_w4_overlap, view(P.ENDING_DIAGONAL, DIAGONAL_PTS, DIAGONAL_TIMES), rc, True)
    _check(R.rule_diagonal_w4_overlap, view(P.ENDING_DIAGONAL, [100, 120, 106, 130, 121]), rc, False)


def test_diagonal_shape(rc) -> None:
    _check(R.rule_diagonal_shape, view(P.ENDING_DIAGONAL, DIAGONAL_PTS, DIAGONAL_TIMES), rc, True)
    expanding = [100, 110, 104, 118, 108, 126]
    _check(R.rule_diagonal_shape, view(P.LEADING_DIAGONAL, expanding), rc, True)
    mixed = [100, 120, 106, 122, 112, 140]  # W3 < W1 but W5 > W3
    _check(R.rule_diagonal_shape, view(P.ENDING_DIAGONAL, mixed), rc, False)


def test_diagonal_trendlines(rc) -> None:
    _check(R.rule_diagonal_trendlines, view(P.ENDING_DIAGONAL, DIAGONAL_PTS, DIAGONAL_TIMES), rc, True)
    # contracting lengths but lower line falls -> lines diverge
    _check(R.rule_diagonal_trendlines, view(P.ENDING_DIAGONAL, [100, 120, 110, 125, 108], [0, 10, 12, 40, 70]), rc, False)


# --- zigzag -----------------------------------------------------------------

def test_zigzag_b(rc) -> None:
    _check(R.rule_zigzag_b_max_retrace, view(P.ZIGZAG, ZIGZAG_PTS), rc, True)
    _check(R.rule_zigzag_b_max_retrace, view(P.ZIGZAG, [100, 120, 98]), rc, False)


def test_zigzag_c(rc) -> None:
    _check(R.rule_zigzag_c_beyond_a, view(P.ZIGZAG, ZIGZAG_PTS), rc, True)
    _check(R.rule_zigzag_c_beyond_a, view(P.ZIGZAG, [100, 120, 110, 119]), rc, False)
    assert R.rule_zigzag_c_beyond_a(view(P.ZIGZAG, [100, 120, 110, 119], last_open=True), rc).deferred


def test_triple_zigzag(rc) -> None:
    ok = view(P.TRIPLE_ZIGZAG, [100, 120, 110, 130, 122, 140])
    _check(R.rule_tz_x2_max_retrace, ok, rc, True)
    _check(R.rule_tz_z_beyond_y, ok, rc, True)
    _check(R.rule_tz_x2_max_retrace, view(P.TRIPLE_ZIGZAG, [100, 120, 110, 130, 108, 140]), rc, False)
    _check(R.rule_tz_z_beyond_y, view(P.TRIPLE_ZIGZAG, [100, 120, 110, 130, 122, 128]), rc, False)


# --- flats ------------------------------------------------------------------

def test_flat_b_min(rc) -> None:
    _check(R.rule_flat_b_min_retrace, view(P.FLAT_REGULAR, FLAT_REG_PTS), rc, True)
    _check(R.rule_flat_b_min_retrace, view(P.FLAT_REGULAR, [100, 120, 106]), rc, False)
    assert R.rule_flat_b_min_retrace(view(P.FLAT_REGULAR, [100, 120, 106], last_open=True), rc).deferred


def test_flat_b_max(rc) -> None:
    _check(R.rule_flat_b_max_retrace, view(P.FLAT_EXPANDED, FLAT_EXP_PTS), rc, True)
    _check(R.rule_flat_b_max_retrace, view(P.FLAT_EXPANDED, [100, 110, 70]), rc, False)


def test_flat_regular(rc) -> None:
    _check(R.rule_flat_regular_b, view(P.FLAT_REGULAR, FLAT_REG_PTS), rc, True)
    _check(R.rule_flat_regular_b, view(P.FLAT_REGULAR, FLAT_EXP_PTS), rc, False)
    _check(R.rule_flat_regular_c, view(P.FLAT_REGULAR, FLAT_REG_PTS), rc, True)
    _check(R.rule_flat_regular_c, view(P.FLAT_REGULAR, [100, 120, 101, 112]), rc, False)


def test_flat_expanded(rc) -> None:
    _check(R.rule_flat_irregular_b, view(P.FLAT_EXPANDED, FLAT_EXP_PTS), rc, True)
    _check(R.rule_flat_irregular_b, view(P.FLAT_EXPANDED, FLAT_REG_PTS), rc, False)
    _check(R.rule_flat_expanded_c, view(P.FLAT_EXPANDED, FLAT_EXP_PTS), rc, True)
    _check(R.rule_flat_expanded_c, view(P.FLAT_EXPANDED, FLAT_RUN_PTS), rc, False)


def test_flat_running(rc) -> None:
    _check(R.rule_flat_running_c, view(P.FLAT_RUNNING, FLAT_RUN_PTS), rc, True)
    _check(R.rule_flat_running_c, view(P.FLAT_RUNNING, FLAT_EXP_PTS), rc, False)
    _check(R.rule_flat_running_c, view(P.FLAT_RUNNING, [100, 120, 95.28, 99]), rc, False)  # C too short


# --- triangles --------------------------------------------------------------

def test_triangle_contracting(rc) -> None:
    _check(R.rule_triangle_contracting, view(P.TRIANGLE_CONTRACTING, TRIANGLE_PTS, TRIANGLE_TIMES), rc, True)
    _check(R.rule_triangle_contracting, view(P.TRIANGLE_CONTRACTING, [100, 120, 104, 126]), rc, False)


def test_triangle_barrier(rc) -> None:
    barrier = [100, 120, 104, 119.8, 110, 118]  # A–C horizontal, B–D rising
    _check(R.rule_triangle_barrier, view(P.TRIANGLE_BARRIER, barrier), rc, True)
    _check(R.rule_triangle_barrier, view(P.TRIANGLE_BARRIER, TRIANGLE_PTS, TRIANGLE_TIMES), rc, False)


def test_triangle_expanding(rc) -> None:
    expanding = [100, 110, 104, 116, 98, 124]
    _check(R.rule_triangle_expanding, view(P.TRIANGLE_EXPANDING, expanding), rc, True)
    _check(R.rule_triangle_expanding, view(P.TRIANGLE_EXPANDING, TRIANGLE_PTS, TRIANGLE_TIMES), rc, False)


# --- combinations -----------------------------------------------------------

def test_combination_x_max(rc) -> None:
    _check(R.rule_combination_x_max, view(P.COMBINATION_WXY, [100, 120, 110, 121]), rc, True)
    _check(R.rule_combination_x_max, view(P.COMBINATION_WXY, [100, 110, 80]), rc, False)


def test_combination_sideways(rc) -> None:
    _check(R.rule_combination_sideways, view(P.COMBINATION_WXY, [100, 120, 110, 121]), rc, True)
    _check(R.rule_combination_sideways, view(P.COMBINATION_WXY, [100, 120, 110, 140]), rc, False)


def test_combination_triangles(rc) -> None:
    last = view(P.COMBINATION_WXY, [100, 120, 110, 121], subs=(P.ZIGZAG, P.ZIGZAG, P.TRIANGLE_CONTRACTING))
    _check(R.rule_combination_triangles, last, rc, True)
    first = view(P.COMBINATION_WXY, [100, 120, 110, 121], subs=(P.TRIANGLE_CONTRACTING, P.ZIGZAG, P.FLAT_REGULAR))
    _check(R.rule_combination_triangles, first, rc, False)


def test_registry_complete() -> None:
    funcs = {name for name in dir(R) if name.startswith("rule_")}
    registered = {r.func.__name__ for r in R.ALL_RULES.values()}
    assert funcs == registered
