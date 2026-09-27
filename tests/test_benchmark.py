"""Performance: 2.000 candles with ~60 pivots on the search degree in < 10 s."""

from __future__ import annotations

import time

import pytest

from elliott.data.loader import frame_to_market_data
from elliott.pipeline import analyze
from tests.synthetic import random_walk


@pytest.mark.benchmark
@pytest.mark.parametrize("seed", [0, 1])
def test_benchmark_2000_candles(settings, seed) -> None:
    data = frame_to_market_data(random_walk(2000, seed=seed), settings.data)
    t0 = time.perf_counter()
    res = analyze(data, settings)
    elapsed = time.perf_counter() - t0
    n_search = res.meta.levels[res.meta.search_level].pivot_count
    print(f"\nBenchmark seed={seed}: {elapsed:.2f} s, search-degree pivots={n_search}, states={res.meta.states_explored}")
    assert 30 <= n_search <= 90
    assert res.scenarios
    assert elapsed < 10.0
    assert not res.meta.budget_exhausted
