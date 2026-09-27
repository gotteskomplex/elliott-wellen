"""Search: enumeration, pruning, incomplete patterns, de-duplication."""

from __future__ import annotations

import pytest

from elliott.guidelines import GuideContext
from elliott.measure import PriceScale
from elliott.models import PatternType as P, SubwaveStatus
from elliott.patterns import get_spec
from elliott.pivots.zigzag import build_hierarchy
from elliott.rules import evaluate_rules
from elliott.search import SearchEngine, dedupe
from elliott.settings import with_overrides
from tests.synthetic import make_pattern


def _engine(settings, syn_df):
    h = build_hierarchy(syn_df, settings.pivots, settings.indicators)
    return SearchEngine(h, PriceScale(False), settings, GuideContext())


def test_fixed_end_enumeration_finds_impulse(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    eng = _engine(settings, syn.df)
    arr = eng.arrays[1]
    start = arr.pos_of_bar[(syn.top_pivots[0][0], -1)]
    end = arr.pos_of_bar[(syn.top_pivots[-1][0], 1)]
    cands = eng.enumerate(1, start, [P.IMPULSE], end=end)
    truth = [b for b, _ in syn.top_pivots]
    assert any(list(c.view.t) == truth for c in cands)
    assert all(c.positions[0] == start and c.positions[-1] == end for c in cands)


def test_all_candidates_satisfy_rules(settings) -> None:
    syn = make_pattern("impulse", bars=300, noise=0.01, seed=5)
    eng = _engine(settings, syn.df)
    _, cands = eng.search()
    assert cands
    for c in cands:
        for rule, res in evaluate_rules(c.spec.all_rules, c.view, settings.rules):
            assert res.passed, f"{c.spec.type}: {rule.key}: {res.reason}"


def test_counts_end_at_right_edge(settings) -> None:
    syn = make_pattern("zigzag", bars=240, noise=0.005, seed=2)
    eng = _engine(settings, syn.df)
    lvl, cands = eng.search()
    last = eng.arrays[lvl].last
    for c in cands:
        assert c.view.last_open
        assert c.positions[-1] == (last - 1 if c.pullback else last)


@pytest.mark.parametrize(("frac", "running"), [(0.5, False), (0.9, True)])
def test_incomplete_impulse_in_wave3(settings, frac, running) -> None:
    """Mid-wave-3 data: the count '1, 2 complete, 3 in progress' is found."""
    syn = make_pattern("impulse", bars=300)
    w2_end, w3_end = syn.top_pivots[2][0], syn.top_pivots[3][0]
    cut = w2_end + int(frac * (w3_end - w2_end))
    eng = _engine(settings, syn.df.iloc[: cut + 1])
    _, cands = eng.search()
    truth = [b for b, _ in syn.top_pivots[:3]]
    hits = [c for c in cands if c.spec.type is P.IMPULSE and c.k == 3 and list(c.view.t[:3]) == truth]
    assert hits, [(c.spec.type, c.k, c.view.t) for c in cands[:10]]
    c = hits[0]
    assert c.view.last_open and c.view.is_open(3)
    assert c.view.running is running  # sub-wave (iv) pullback -> wave 3 still running


@pytest.mark.parametrize("wave", [4, 5])
def test_incomplete_impulse_later_waves_rank_high(settings, wave) -> None:
    syn = make_pattern("impulse", bars=300)
    a, b = syn.top_pivots[wave - 1][0], syn.top_pivots[wave][0]
    cut = a + int(0.75 * (b - a))
    eng = _engine(settings, syn.df.iloc[: cut + 1])
    _, cands = eng.search()
    truth = [t for t, _ in syn.top_pivots[:wave]]
    ranks = [i for i, c in enumerate(cands) if c.spec.type is P.IMPULSE and c.k == wave and list(c.view.t[:wave]) == truth]
    assert ranks and ranks[0] < 5


def test_skip_limit_respected(settings) -> None:
    s = with_overrides(settings, {"search": {"max_skip": 1}})
    syn = make_pattern("impulse", bars=300)
    eng = _engine(s, syn.df)
    _, cands = eng.search()
    for c in cands:
        steps = [b - a for a, b in zip(c.positions, c.positions[1:])]
        assert max(steps) <= 3


def test_subwave_validation(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    eng = _engine(settings, syn.df)
    _, cands = eng.search()
    best = cands[0]
    assert best.spec.type is P.IMPULSE
    statuses = [info.status for info in best.subwaves[:4]]
    assert all(s is SubwaveStatus.VALIDATED for s in statuses)
    assert best.subwaves[0].pattern in get_spec(P.IMPULSE).sub_types[0]
    assert best.subwaves[1].pattern is not None and best.subwaves[1].pattern.value.startswith(("zigzag", "double", "flat", "comb"))


def test_subwave_not_validated_without_fine_pivots(settings) -> None:
    syn = make_pattern("impulse", bars=300, depth=1)  # no subwaves at all
    eng = _engine(settings, syn.df)
    _, cands = eng.search()
    imp = next(c for c in cands if c.spec.type is P.IMPULSE and c.k == 5)
    assert all(info.status is SubwaveStatus.NOT_VALIDATED for info in imp.subwaves)


def test_motive_wave_with_three_legs_fails(settings) -> None:
    syn = make_pattern("zigzag", bars=240)
    eng = _engine(settings, syn.df)
    arr = eng.arrays[-1]
    a = (syn.top_pivots[1][0], 1)
    b = (syn.top_pivots[2][0], -1)  # wave B (3 legs) tested as motive
    info = eng.validate_wave(len(eng.arrays) - 1, a, b, frozenset({P.IMPULSE}), 1)
    assert info.status is SubwaveStatus.FAILED
    assert arr is not None


def test_dedupe_family(settings) -> None:
    syn = make_pattern("ending_diagonal", bars=240)
    eng = _engine(settings, syn.df)
    _, cands = eng.search()
    keys = [(c.spec.family, c.positions) for c in cands]
    assert len(keys) == len(set(keys))
    again = dedupe(cands + cands, settings)
    assert len(again) == len(cands)


def test_time_budget_flag(settings) -> None:
    s = with_overrides(settings, {"search": {"time_budget_s": 1e-6}})
    syn = make_pattern("impulse", bars=300, noise=0.01)
    eng = _engine(s, syn.df)
    eng.search()
    assert eng.stats.budget_exhausted


def test_user_start_point(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    eng = _engine(settings, syn.df)
    start_bar = syn.top_pivots[2][0]
    _, cands = eng.search(start_bar=start_bar)
    assert cands and all(c.view.t[0] == start_bar for c in cands)


@pytest.mark.parametrize("name", ["impulse", "zigzag", "flat_expanded", "triangle", "ending_diagonal"])
def test_prior_move_known(settings, name) -> None:
    syn = make_pattern(name, bars=240, lead_in=0)
    eng = _engine(settings, syn.df)
    arr = eng.arrays[1]
    assert eng.prior_move(1, 0) is None  # nothing before the first pivot
    assert eng.prior_move(1, 2) is not None and eng.prior_move(1, 2) > 0
    assert len(arr) > 3
