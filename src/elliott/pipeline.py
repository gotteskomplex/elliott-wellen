"""Single entry point of the analysis core: ``analyze(data, settings)``.

This module (like everything in the core) knows nothing about Streamlit or
Plotly; the CLI, the app and the reports only consume the returned
:class:`~elliott.models.AnalysisResult`.
"""

from __future__ import annotations

import time
from datetime import datetime

import numpy as np
import pandas as pd

from elliott import DISCLAIMER
from elliott.data.indicators import rsi
from elliott.data.loader import MarketData, resolve_log_scale
from elliott.fmt import fmt_price
from elliott.guidelines import GuideContext
from elliott.labels import display_label
from elliott.measure import PriceScale
from elliott.models import (
    AnalysisMeta, AnalysisResult, Direction, LevelInfo, RuleCheck, Scenario, SubwaveStatus, Wave, WaveCount,
)
from elliott.pivots.zigzag import build_hierarchy
from elliott.projection import is_invalidated_at, nudge_beyond, project
from elliott.rules import evaluate_rules
from elliott.scoring import softmax
from elliott.search import Candidate, SearchEngine
from elliott.settings import Settings


def candidate_to_count(cand: Candidate, engine: SearchEngine, degree: int, cfg: Settings, max_depth: int = 3) -> WaveCount:
    """Convert a search candidate (and its validated subcounts) into a WaveCount."""
    arr = engine.arrays[cand.level]
    spec, v = cand.spec, cand.view
    waves: list[Wave] = []
    for i in range(1, cand.k + 1):
        a, b = cand.positions[i - 1], cand.positions[i]
        start, end = arr.pivots[a], arr.pivots[b]
        ratios: dict[str, float] = {"vs_w1": v.L(i) / v.L(1)}
        if i >= 2:
            ratios["vs_prev"] = v.L(i) / v.L(i - 1)
        if i >= 3:
            ratios["vs_prev_same_dir"] = v.L(i) / v.L(i - 2)
        info = cand.subwaves[i - 1] if cand.subwaves else None
        subcount = None
        if info is not None and info.candidate is not None and max_depth > 0:
            subcount = candidate_to_count(info.candidate, engine, degree + 1, cfg, max_depth - 1)
        price_dir = (1 if v.wave_up(i) else -1) * v.d
        waves.append(
            Wave(
                label=spec.labels[i - 1],
                display_label=display_label(spec, i, degree, cfg.plotting),
                start=start,
                end=end,
                direction=Direction.from_sign(price_dir),
                wave_class=spec.wave_classes[i - 1],
                length_price=abs(end.price - start.price),
                length_measure=v.L(i),
                duration_bars=max(0, end.idx - start.idx),
                ratios=ratios,
                open=v.is_open(i),
                subwave_status=info.status if info else SubwaveStatus.NOT_VALIDATED,
                subwave_note=info.note if info else "",
                subcount=subcount,
            )
        )
    return WaveCount(
        pattern=spec.type,
        direction=Direction.from_sign(v.d),
        degree=degree,
        waves=waves,
        expected_waves=spec.n,
        complete=cand.k == spec.n,
    )


def _notes(cand: Candidate) -> list[str]:
    notes: list[str] = []
    v = cand.view
    if cand.score is not None:
        for p in cand.score.penalties:
            if p.name != "complexity":
                notes.append(p.reason)
    if v.running:
        notes.append(f"Welle {cand.spec.labels[cand.k - 1]} läuft noch (letzte Pivots liegen innerhalb der Welle)")
    if cand.subwaves:
        failed = [cand.spec.labels[i] for i, s in enumerate(cand.subwaves) if s.status is SubwaveStatus.FAILED]
        if failed:
            notes.append("Binnenstruktur nicht bestätigt für Welle(n) " + ", ".join(failed))
        unknown = [cand.spec.labels[i] for i, s in enumerate(cand.subwaves) if s.status is SubwaveStatus.NOT_VALIDATED]
        if unknown:
            notes.append("Nicht validiert (keine feineren Daten oder Welle läuft): " + ", ".join(unknown))
    return notes


def analyze(
    data: MarketData,
    cfg: Settings,
    start: datetime | pd.Timestamp | None = None,
    top_n: int | None = None,
    log_mode: str | None = None,
) -> AnalysisResult:
    """Run the complete analysis.

    Args:
        data: validated market data.
        cfg: settings.
        start: optional start date for the count (nearest pivot is used).
        top_n: number of scenarios that receive a softmax weight.
        log_mode: override of ``data.log_mode`` (auto/linear/log).
    """
    t0 = time.perf_counter()
    df = data.df
    warnings = list(data.warnings)
    log = resolve_log_scale(df, log_mode or cfg.data.log_mode, cfg.data.log_auto_ratio)
    scale = PriceScale(log)
    hierarchy = build_hierarchy(df, cfg.pivots, cfg.indicators)
    volume = df["volume"].to_numpy(dtype=float) if data.has_volume else None
    ctx = GuideContext(volume=volume, rsi=rsi(df["close"], cfg.indicators.rsi_period).to_numpy(dtype=float))
    engine = SearchEngine(hierarchy, scale, cfg, ctx)

    start_bar = None
    if start is not None:
        ts = pd.Timestamp(start)
        start_bar = int(np.argmin(np.abs((df.index - ts).total_seconds())))
    level, cands = engine.search(start_bar=start_bar)
    if not any(hierarchy.levels):
        warnings.append("Keine Wendepunkte gefunden – ZigZag-Schwelle verkleinern.")
    elif not cands:
        warnings.append("Keine regelkonforme Zählung gefunden – andere Schwelle oder anderen Startpunkt versuchen.")
    if engine.stats.budget_exhausted:
        warnings.append("Suchzeit-Budget erreicht – Ergebnisse sind möglicherweise unvollständig.")

    degree = cfg.plotting.top_degree
    scenarios: list[Scenario] = []
    for rank, cand in enumerate(cands, start=1):
        proj, sub = project(cand, engine, df, scale, cfg, degree)
        if sub is not None:
            proj.current_subcount = candidate_to_count(sub, engine, degree + 1, cfg, max_depth=1)
        checks = [
            RuleCheck(name=rule.key, passed=res.passed, reason=res.reason, deferred=res.deferred)
            for rule, res in evaluate_rules(cand.spec.all_rules, cand.view, cfg.rules)
        ]
        assert cand.score is not None
        scenarios.append(
            Scenario(
                id=f"S{rank}",
                rank=rank,
                count=candidate_to_count(cand, engine, degree, cfg),
                score=cand.score,
                projection=proj,
                rule_checks=checks,
                notes=_notes(cand),
                start_pivot_level=cand.level,
            )
        )

    n_top = top_n or cfg.scoring.top_n
    weights = softmax([s.score.total for s in scenarios[:n_top]], cfg.scoring.softmax_temperature)
    for s, w in zip(scenarios, weights):
        s.probability = w
    _link_alternatives(scenarios)

    meta = AnalysisMeta(
        source=data.source,
        start=df.index[0].to_pydatetime(),
        end=df.index[-1].to_pydatetime(),
        bars=len(df),
        log_scale=log,
        last_close=float(df["close"].iloc[-1]),
        has_volume=data.has_volume,
        levels=[
            LevelInfo(level=i, name=hierarchy.names[i], threshold=hierarchy.thresholds[i].describe(), pivot_count=len(p))
            for i, p in enumerate(hierarchy.levels)
        ],
        search_level=level,
        runtime_s=time.perf_counter() - t0,
        states_explored=engine.stats.states,
        budget_exhausted=engine.stats.budget_exhausted,
    )
    return AnalysisResult(
        meta=meta,
        pivots={i: p for i, p in enumerate(hierarchy.levels)},
        scenarios=scenarios,
        warnings=warnings,
        disclaimer=DISCLAIMER,
    )


def _link_alternatives(scenarios: list[Scenario]) -> None:
    """For every scenario: which other scenario survives its invalidation?"""
    for s in scenarios:
        if s.projection is None:
            continue
        inv = s.projection.invalidation
        price = nudge_beyond(inv)
        alt = next(
            (
                o for o in scenarios
                if o is not s and o.projection is not None and not is_invalidated_at(o.projection.invalidations, price)
            ),
            None,
        )
        where = f"{'unter' if inv.side == 'below' else 'über'} {fmt_price(inv.price)}"
        if alt is None:
            s.projection.alternative = (
                f"Wird das Szenario {where} invalidiert, bleibt keine der gefundenen Zählungen gültig – "
                "Neubewertung auf höherem Grad nötig."
            )
        else:
            s.projection.alternative_scenario_id = alt.id
            s.projection.alternative = (
                f"Bei Invalidierung ({where}) wird Szenario {alt.id} favorisiert: "
                f"{alt.count.pattern.german} – {alt.projection.position if alt.projection else ''}"
            )
