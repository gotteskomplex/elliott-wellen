"""End-to-end: correct count ranking, hard-rule invariant, edge cases, layering."""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from elliott.data.loader import DataValidationError, frame_to_market_data
from elliott.measure import make_view
from elliott.models import PatternType as P
from elliott.patterns import get_spec
from elliott.pipeline import analyze
from elliott.rules import evaluate_rules
from elliott.settings import load_settings
from tests.synthetic import TEMPLATES, make_pattern, random_walk, sideways

FAMILY = {
    "impulse": "impulse", "impulse_ext3": "impulse", "impulse_truncated": "impulse", "zigzag": "zigzag",
    "flat_regular": "flat", "flat_expanded": "flat", "triangle": "triangle", "ending_diagonal": "diagonal",
}


def _truth_rank(result, syn, family: str, tol: int = 4) -> int | None:
    truth = [b for b, _ in syn.top_pivots]
    for s in result.scenarios:
        bars = [s.count.waves[0].start.idx] + [w.end.idx for w in s.count.waves]
        if get_spec(s.count.pattern).family == family and len(bars) == len(truth) and all(
            abs(a - b) <= tol for a, b in zip(bars, truth)
        ):
            return s.rank
    return None


def _run(settings, syn):
    return analyze(frame_to_market_data(syn.df, settings.data), settings)


def test_textbook_impulse_rank_1(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    assert _truth_rank(_run(settings, syn), syn, "impulse", tol=0) == 1


@pytest.mark.parametrize("seed", [1, 2, 3, 4, 5])
def test_textbook_impulse_with_moderate_noise_top3(settings, seed) -> None:
    syn = make_pattern("impulse", bars=300, noise=0.01, seed=seed)
    rank = _truth_rank(_run(settings, syn), syn, "impulse")
    assert rank is not None and rank <= 3


@pytest.mark.parametrize("name", sorted(TEMPLATES))
def test_synthetic_patterns_top3(settings, name) -> None:
    prefix = 0.0 if name.startswith(("impulse", "ending")) else 2.0
    syn = make_pattern(name, bars=300, noise=0.005, seed=7, prefix=prefix)
    rank = _truth_rank(_run(settings, syn), syn, FAMILY[name])
    assert rank is not None and rank <= 3, rank


def test_down_trend_impulse(settings) -> None:
    syn = make_pattern("impulse", bars=300, direction=-1, start_price=200.0, height=60.0)
    res = _run(settings, syn)
    assert _truth_rank(res, syn, "impulse", tol=0) == 1
    assert res.scenarios[0].count.direction.value == "down"


def test_log_scale_long_trend(settings) -> None:
    syn = make_pattern("impulse", bars=400, start_price=10.0, height=90.0)
    res = _run(settings, syn)
    assert res.meta.log_scale  # max/min > 3 -> automatic log
    assert res.scenarios


@pytest.mark.parametrize(
    "df_factory",
    [
        lambda: make_pattern("impulse", bars=300, noise=0.02, seed=11).df,
        lambda: make_pattern("flat_expanded", bars=300, noise=0.01, seed=3, prefix=2.0).df,
        lambda: sideways(400, seed=2),
        lambda: random_walk(1500, seed=5),
    ],
)
def test_no_output_violates_a_hard_rule(settings, df_factory) -> None:
    """Acceptance criterion: every reported count is re-checked against rules.py."""
    res = analyze(frame_to_market_data(df_factory(), settings.data), settings)
    for s in res.scenarios:
        running = s.projection is not None and s.projection.current_wave_running
        _assert_count_valid(s.count, settings, res.meta.log_scale, last_open=True, running=running)
        for w in s.count.waves:
            if w.subcount is not None:
                _assert_count_valid(w.subcount, settings, res.meta.log_scale, last_open=False)


def _assert_count_valid(count, settings, log: bool, last_open: bool, running: bool = False) -> None:
    """Rebuild the count from its output pivots and evaluate ALL hard rules."""
    import math

    spec = get_spec(count.pattern)
    pts = [count.waves[0].start] + [w.end for w in count.waves]
    d = 1 if count.direction.value == "up" else -1
    tp = [math.log(p.price) if log else p.price for p in pts]
    subs = tuple(w.subcount.pattern if w.subcount else None for w in count.waves)
    v = make_view(count.pattern, spec.n, d, tp, [p.idx for p in pts], [p.price for p in pts], last_open, subs)
    v.running = running
    failed = [(r.key, res.reason) for r, res in evaluate_rules(spec.all_rules, v, settings.rules) if not res.passed]
    assert not failed, (count.pattern, failed)


def test_all_rules_checked_in_scenario_output(settings) -> None:
    syn = make_pattern("impulse", bars=300, noise=0.01, seed=3)
    res = _run(settings, syn)
    for s in res.scenarios:
        assert all(rc.passed for rc in s.rule_checks), s.id


# --- edge cases ---------------------------------------------------------------


def test_very_few_bars(settings) -> None:
    syn = make_pattern("impulse", bars=12, depth=1)
    with pytest.raises(DataValidationError):
        frame_to_market_data(syn.df.iloc[:10], settings.data)
    cfg = settings.data.model_copy(update={"min_bars": 5})
    res = analyze(frame_to_market_data(syn.df, cfg), settings)
    assert res.meta.bars == len(syn.df)  # does not crash


def test_sideways_market(settings) -> None:
    res = analyze(frame_to_market_data(sideways(300, noise=0.004, seed=1), settings.data), settings)
    assert res.meta.bars == 300
    for s in res.scenarios:
        assert s.projection is not None and s.projection.invalidation.reason


def test_single_extreme_spike(settings) -> None:
    syn = make_pattern("impulse", bars=300, noise=0.005, seed=2)
    df = syn.df.copy()
    i = 150
    df.iloc[i, df.columns.get_loc("high")] = df["high"].iloc[i] * 1.5
    res = analyze(frame_to_market_data(df, settings.data), settings)
    assert res.scenarios  # still produces scenarios
    for s in res.scenarios:
        assert all(rc.passed for rc in s.rule_checks)


def test_missing_volume(settings) -> None:
    syn = make_pattern("impulse", bars=300, with_volume=False)
    res = analyze(frame_to_market_data(syn.df, settings.data), settings)
    assert not res.meta.has_volume
    vol = next(g for g in res.scenarios[0].score.guidelines if g.name == "impulse_volume")
    assert vol.value is None


def test_flat_line_no_pivots(settings) -> None:
    idx = pd.date_range("2024-01-01", periods=60, freq="D")
    df = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0}, index=idx)
    res = analyze(frame_to_market_data(df, settings.data), settings)
    assert res.scenarios == [] and res.warnings


def test_user_start_date(settings) -> None:
    syn = make_pattern("impulse", bars=300)
    start = syn.df.index[syn.top_pivots[2][0]]
    res = analyze(frame_to_market_data(syn.df, settings.data), settings, start=start)
    assert res.scenarios and all(s.count.waves[0].start.idx == syn.top_pivots[2][0] for s in res.scenarios)


def test_core_has_no_ui_dependencies() -> None:
    """The analysis core must not import Streamlit or Plotly."""
    code = (
        "import sys; import elliott.pipeline, elliott.search, elliott.projection, elliott.report; "
        "bad=[m for m in ('streamlit','plotly') if m in sys.modules]; print(','.join(bad))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == ""
