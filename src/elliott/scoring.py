"""Score aggregation (0–100), penalties, Occam complexity and softmax weights."""

from __future__ import annotations

import math
from collections.abc import Sequence

from elliott.guidelines import GuideContext
from elliott.measure import CountView
from elliott.models import GuidelineScore, Penalty, ScoreBreakdown
from elliott.patterns.base import PatternSpec
from elliott.settings import Settings


def _penalties(spec: PatternSpec, v: CountView, cfg: Settings) -> list[Penalty]:
    pens = cfg.guidelines.penalties
    out: list[Penalty] = []
    if spec.family in ("impulse", "diagonal") and v.has(5) and not v.is_open(5) and v.x[5] <= v.x[3]:
        out.append(Penalty(name="truncation", points=pens["truncation"], reason="Verkürzte Welle 5 (Truncation): W5 übertrifft das Ende von W3 nicht"))
    if spec.penalty is not None:
        out.append(Penalty(name=spec.penalty, points=pens[spec.penalty], reason=f"Seltenes Muster: {spec.type.german}"))
    if spec.family == "diagonal" and v.has(3) and v.L(3) > v.L(1):
        out.append(Penalty(name="expanding_diagonal", points=pens["expanding_diagonal"], reason="Seltenes Muster: expandierendes Diagonal"))
    return out


def complexity_of(spec: PatternSpec, cfg: Settings) -> float:
    """Occam complexity of a pattern (from the config)."""
    return float(cfg.guidelines.complexity.by_pattern.get(spec.type.value, 0.0))


def _evaluate(spec: PatternSpec, v: CountView, ctx: GuideContext, cfg: Settings):  # type: ignore[no-untyped-def]
    """Yield ``(guideline, weight, value_or_None, reason, pending)``.

    Open last wave (ends at the provisional right-edge pivot):
    * known to be *running* (pullback inside the wave or a deferred rule): its
      current length says nothing, guidelines depending on it are pending;
    * otherwise it may or may not be finished: a guideline that depends on it
      gets the benefit of the doubt – ``max(current value, pending value)``.
    """
    pending_value = cfg.guidelines.p("pending_value")
    head = v.head(v.k - 1) if v.last_open and v.k > 1 else None
    for g in spec.all_guidelines:
        w = cfg.guidelines.w(g.key)
        if g.key == "completeness" or head is None:
            res = g(v, ctx, cfg)
        else:
            res_head = g(head, ctx, cfg)
            depends = res_head is not None and res_head.pending
            if not depends:
                res = g(v, ctx, cfg) if not v.running else res_head
            elif v.running:
                res = res_head
            else:
                res_full = g(v, ctx, cfg)
                if res_full is not None and res_full.value is not None and res_full.value < pending_value:
                    yield g, w, pending_value, f"{res_full.reason} – letzte Welle offen, neutral bewertet", False
                    continue
                res = res_full
        if res is None:
            yield g, w, None, "nicht anwendbar", False
        elif res.pending:
            yield g, w, pending_value, res.reason, True
        else:
            yield g, w, res.value, res.reason, False


def quick_score(spec: PatternSpec, v: CountView, ctx: GuideContext, cfg: Settings) -> float:
    """Fast total score without building report objects (used inside the search)."""
    num = den = 0.0
    for _, w, value, _, _ in _evaluate(spec, v, ctx, cfg):
        if value is None:
            continue
        num += w * value
        den += w
    base = 100.0 * num / den if den > 0 else 0.0
    pen = sum(p.points for p in _penalties(spec, v, cfg))
    comp = cfg.guidelines.complexity.weight * complexity_of(spec, cfg)
    return max(0.0, min(100.0, base - pen - comp))


def score_view(spec: PatternSpec, v: CountView, ctx: GuideContext, cfg: Settings) -> ScoreBreakdown:
    """Full score with a per-guideline breakdown (points + German reasons)."""
    items: list[GuidelineScore] = []
    num = den = 0.0
    for g, w, value, reason, _pending in _evaluate(spec, v, ctx, cfg):
        items.append(GuidelineScore(name=g.key, title=g.title, weight=w, value=value, points=0.0, reason=reason))
        if value is not None:
            num += w * value
            den += w
    base = 100.0 * num / den if den > 0 else 0.0
    # Points: each guideline's share of the 0–100 guideline score.
    for it in items:
        if it.value is not None and den > 0:
            it.points = 100.0 * it.weight * it.value / den
    penalties = _penalties(spec, v, cfg)
    comp = cfg.guidelines.complexity.weight * complexity_of(spec, cfg)
    if comp > 0:
        penalties.append(Penalty(name="complexity", points=comp, reason=f"Komplexität (Occam) von {spec.type.german}"))
    total = max(0.0, min(100.0, base - sum(p.points for p in penalties)))
    return ScoreBreakdown(total=total, guideline_score=base, guidelines=items, penalties=penalties, complexity=comp)


def softmax(scores: Sequence[float], temperature: float) -> list[float]:
    """Heuristic relative weights from scores (NOT a statistical probability)."""
    if not scores:
        return []
    m = max(scores)
    exps = [math.exp((s - m) / temperature) for s in scores]
    total = sum(exps)
    return [e / total for e in exps]
