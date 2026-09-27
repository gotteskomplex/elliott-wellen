"""Soft Elliott guidelines (scoring).

Each guideline returns a value in ``[0, 1]`` (or ``None`` if not applicable)
plus a German explanation. Weights and all target ratios live in the config
(``guidelines.weights`` / ``guidelines.params``). Fibonacci hits are scored with
tolerance bands and a continuous decay (see :func:`elliott.fibonacci.band_score`).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from elliott.fibonacci import fib_score, range_score
from elliott.fmt import fmt_pct, fmt_ratio
from elliott.measure import CountView
from elliott.models import PATTERN_LABELS, PatternType
from elliott.settings import Settings

SHARP = frozenset({PatternType.ZIGZAG, PatternType.DOUBLE_ZIGZAG, PatternType.TRIPLE_ZIGZAG})


@dataclass(frozen=True, slots=True)
class GuidelineResult:
    value: float | None
    reason: str


@dataclass
class GuideContext:
    """Market data needed by some guidelines (volume, momentum, subwaves)."""

    volume: np.ndarray | None = None
    rsi: np.ndarray | None = None
    subwave_values: tuple[float | None, ...] | None = None
    subwave_notes: tuple[str, ...] = field(default_factory=tuple)


GuidelineFn = Callable[[CountView, GuideContext, Settings], GuidelineResult | None]


@dataclass(frozen=True, slots=True)
class Guideline:
    key: str
    title: str
    func: GuidelineFn

    def __call__(self, v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult | None:
        return self.func(v, ctx, cfg)


def _na(reason: str) -> GuidelineResult:
    return GuidelineResult(None, reason)


def _lab(v: CountView, wave: int) -> str:
    return PATTERN_LABELS[v.pattern][wave - 1]


def _targets_text(targets: list[float]) -> str:
    return " / ".join(fmt_ratio(t) for t in targets)


# ---------------------------------------------------------------------------
# impulse
# ---------------------------------------------------------------------------


def g_impulse_w2_retrace(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(2):
        return _na("Welle 2 fehlt")
    lo, hi = cfg.guidelines.p("impulse_w2_range")
    r = v.L(2) / v.L(1)
    s = range_score(r, lo, hi, cfg.fib)
    return GuidelineResult(s, f"W2 retraced {fmt_pct(r)} von W1 (typisch {fmt_pct(lo)}–{fmt_pct(hi)})")


def g_impulse_w4_retrace(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(4):
        return _na("Welle 4 fehlt")
    lo, hi = cfg.guidelines.p("impulse_w4_range")
    r = v.L(4) / v.L(3)
    s = range_score(r, lo, hi, cfg.fib)
    return GuidelineResult(s, f"W4 retraced {fmt_pct(r)} von W3 (typisch {fmt_pct(lo)}–{fmt_pct(hi)})")


def g_impulse_alternation(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(4):
        return _na("Welle 4 fehlt")
    p = cfg.guidelines.params
    r2, r4 = v.L(2) / v.L(1), v.L(4) / v.L(3)
    parts: list[float] = [min(1.0, abs(r2 - r4) / p["alternation_depth_scale"])]
    text = [f"Tiefe W2 {fmt_pct(r2)} vs. W4 {fmt_pct(r4)}"]
    d2, d4 = v.dur(2), v.dur(4)
    time_ratio = max(d2, d4) / min(d2, d4)
    parts.append(min(1.0, max(0.0, (time_ratio - 1.0) / (p["alternation_time_ratio"] - 1.0))))
    text.append(f"Dauer {d2} vs. {d4} Kerzen")
    subs = v.sub_patterns
    if subs is not None and len(subs) >= 4 and subs[1] is not None and subs[3] is not None:
        differ = (subs[1] in SHARP) != (subs[3] in SHARP)
        parts.append(1.0 if differ else 0.0)
        text.append(f"Form {subs[1].german} vs. {subs[3].german}")
    return GuidelineResult(sum(parts) / len(parts), "Alternation: " + ", ".join(text))


def g_impulse_extension(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(5):
        return _na("Welle 5 fehlt")
    p = cfg.guidelines.params
    lengths = {1: v.L(1), 3: v.L(3), 5: v.L(5)}
    order = sorted(lengths, key=lambda w: lengths[w], reverse=True)
    longest, second = order[0], order[1]
    ext = lengths[longest] / lengths[second] if lengths[second] > 0 else math.inf
    base = min(1.0, max(0.0, (ext - 1.0) / (p["extension_ratio"] - 1.0)))
    factor = 1.0 if longest == 3 else p["extension_non_w3_factor"]
    return GuidelineResult(
        base * factor,
        f"Welle {longest} ist extended ({fmt_ratio(ext)} × zweitlängste Welle)",
    )


def g_impulse_w3_ratio(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(3):
        return _na("Welle 3 fehlt")
    targets = cfg.guidelines.p("impulse_w3_targets")
    r = v.L(3) / v.L(1)
    s, t = fib_score(r, targets, cfg.fib)
    return GuidelineResult(s, f"W3 = {fmt_ratio(r)} × W1 (Ziele {_targets_text(targets)}; nächstes {fmt_ratio(t)})")


def g_impulse_w5_ratio(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(5):
        return _na("Welle 5 fehlt")
    p = cfg.guidelines.params
    r1 = v.L(5) / v.L(1)
    s1, t1 = fib_score(r1, p["impulse_w5_targets_w1"], cfg.fib)
    base = v.x[3] - v.x[0]
    r13 = v.L(5) / base if base > 0 else math.inf
    s2, t2 = fib_score(r13, p["impulse_w5_targets_w1w3"], cfg.fib)
    if s1 >= s2:
        return GuidelineResult(s1, f"W5 = {fmt_ratio(r1)} × W1 (nächstes Ziel {fmt_ratio(t1)})")
    return GuidelineResult(s2, f"W5 = {fmt_ratio(r13)} × (W1 bis W3) (nächstes Ziel {fmt_ratio(t2)})")


def g_impulse_channel(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(5):
        return _na("Welle 5 fehlt")
    if v.t[4] == v.t[2]:
        return _na("Kanal nicht bestimmbar")
    slope = (v.x[4] - v.x[2]) / (v.t[4] - v.t[2])
    channel = v.x[3] + slope * (v.t[5] - v.t[3])
    dev = abs(v.x[5] - channel) / v.L(5)
    sigma = cfg.guidelines.p("channel_sigma")
    s = math.exp(-0.5 * (dev / sigma) ** 2)
    return GuidelineResult(s, f"W5-Ende weicht {fmt_pct(dev)} (von W5) von der Kanal-Parallele durch W3 ab")


def _mean_volume(vol: np.ndarray, a: int, b: int) -> float:
    seg = vol[a : b + 1]
    return float(seg.mean()) if len(seg) else 0.0


def g_impulse_volume(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if ctx.volume is None:
        return _na("Keine Volumendaten – neutral")
    if not v.has(5):
        return _na("Welle 5 fehlt")
    v3 = _mean_volume(ctx.volume, v.t[2], v.t[3])
    v5 = _mean_volume(ctx.volume, v.t[4], v.t[5])
    if v3 <= 0 and v5 <= 0:
        return _na("Kein Volumen in W3/W5")
    s = 1.0 if v3 > v5 else 0.0
    return GuidelineResult(s, f"Ø-Volumen W3 {v3:,.0f} vs. W5 {v5:,.0f}".replace(",", "."))


def g_impulse_momentum(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if ctx.rsi is None:
        return _na("Kein Momentum-Indikator")
    if not v.has(5):
        return _na("Welle 5 fehlt")
    r3, r5 = float(ctx.rsi[v.t[3]]), float(ctx.rsi[v.t[5]])
    min_diff = cfg.guidelines.p("momentum_min_rsi_diff")
    diverges = (r5 < r3 - min_diff) if v.d > 0 else (r5 > r3 + min_diff)
    return GuidelineResult(
        1.0 if diverges else 0.0,
        f"RSI am Ende W3 {r3:.0f} vs. W5 {r5:.0f} – {'Divergenz' if diverges else 'keine Divergenz'}",
    )


def g_impulse_time_w2_w4(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(4):
        return _na("Welle 4 fehlt")
    d2, d4 = v.dur(2), v.dur(4)
    ratio = max(d2, d4) / min(d2, d4)
    limit = cfg.guidelines.p("time_w2_w4_max_ratio")
    s = 1.0 if ratio <= limit else limit / ratio
    return GuidelineResult(s, f"Zeitverhältnis W2/W4 = {fmt_ratio(ratio)} (max. {fmt_ratio(limit)} unauffällig)")


# ---------------------------------------------------------------------------
# diagonal
# ---------------------------------------------------------------------------


def g_diagonal_ratios(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(3):
        return _na("Welle 3 fehlt")
    contracting = v.L(3) < v.L(1)
    targets = cfg.guidelines.p("diagonal_contracting_targets" if contracting else "diagonal_expanding_targets")
    scores = [fib_score(v.L(3) / v.L(1), targets, cfg.fib)[0]]
    if v.has(5):
        scores.append(fib_score(v.L(5) / v.L(3), targets, cfg.fib)[0])
    text = f"W3/W1 = {fmt_ratio(v.L(3) / v.L(1))}" + (f", W5/W3 = {fmt_ratio(v.L(5) / v.L(3))}" if v.has(5) else "")
    return GuidelineResult(sum(scores) / len(scores), f"{text} (Ziele {_targets_text(targets)})")


# ---------------------------------------------------------------------------
# corrections
# ---------------------------------------------------------------------------


def g_zigzag_b_retrace(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(2):
        return _na("B fehlt")
    lo, hi = cfg.guidelines.p("zigzag_b_range")
    r = v.L(2) / v.L(1)
    return GuidelineResult(range_score(r, lo, hi, cfg.fib), f"B retraced {fmt_pct(r)} von A (typisch {fmt_pct(lo)}–{fmt_pct(hi)})")


def g_zigzag_c_ratio(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(3):
        return _na("C fehlt")
    targets = cfg.guidelines.p("zigzag_c_targets")
    r = v.L(3) / v.L(1)
    s, _ = fib_score(r, targets, cfg.fib)
    return GuidelineResult(s, f"C = {fmt_ratio(r)} × A (Ziele {_targets_text(targets)})")


def g_double_zigzag_y_ratio(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(3):
        return _na("Y fehlt")
    targets = cfg.guidelines.p("double_zigzag_y_targets")
    scores = [fib_score(v.L(3) / v.L(1), targets, cfg.fib)[0]]
    text = f"Y = {fmt_ratio(v.L(3) / v.L(1))} × W"
    if v.has(5):
        scores.append(fib_score(v.L(5) / v.L(3), targets, cfg.fib)[0])
        text += f", Z = {fmt_ratio(v.L(5) / v.L(3))} × Y"
    return GuidelineResult(sum(scores) / len(scores), f"{text} (Ziele {_targets_text(targets)})")


def g_flat_b_ratio(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(2):
        return _na("B fehlt")
    r = v.L(2) / v.L(1)
    if v.pattern is PatternType.FLAT_REGULAR:
        targets = cfg.guidelines.p("flat_regular_b_targets")
        s, _ = fib_score(r, targets, cfg.fib)
        return GuidelineResult(s, f"B = {fmt_pct(r)} von A (Regular: ≈ {_targets_text(targets)})")
    lo, hi = cfg.guidelines.p("flat_expanded_b_range")
    return GuidelineResult(range_score(r, lo, hi, cfg.fib), f"B = {fmt_pct(r)} von A (typisch {fmt_pct(lo)}–{fmt_pct(hi)})")


def g_flat_c_ratio(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(3):
        return _na("C fehlt")
    key = {
        PatternType.FLAT_REGULAR: "flat_regular_c_targets",
        PatternType.FLAT_EXPANDED: "flat_expanded_c_targets",
        PatternType.FLAT_RUNNING: "flat_running_c_targets",
    }[v.pattern]
    targets = cfg.guidelines.p(key)
    r = v.L(3) / v.L(1)
    s, _ = fib_score(r, targets, cfg.fib)
    return GuidelineResult(s, f"C = {fmt_ratio(r)} × A (Ziele {_targets_text(targets)})")


def g_triangle_ratios(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(3):
        return _na("C fehlt")
    expanding = v.pattern is PatternType.TRIANGLE_EXPANDING
    targets = cfg.guidelines.p("triangle_expanding_targets" if expanding else "triangle_contracting_targets")
    scores, parts = [], []
    for i in range(3, v.k + 1):
        r = v.L(i) / v.L(i - 2)
        scores.append(fib_score(r, targets, cfg.fib)[0])
        parts.append(f"{_lab(v, i)}/{_lab(v, i - 2)} = {fmt_ratio(r)}")
    return GuidelineResult(sum(scores) / len(scores), ", ".join(parts) + f" (Ziel {_targets_text(targets)})")


def g_combination_sideways(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if not v.has(3):
        return _na("Y fehlt")
    sigma = cfg.guidelines.p("combination_sideways_sigma")
    devs = [abs(v.x[3] - v.x[1]) / v.L(1)]
    if v.has(5):
        devs.append(abs(v.x[5] - v.x[3]) / v.L(3))
    scores = [math.exp(-0.5 * (d / sigma) ** 2) for d in devs]
    return GuidelineResult(
        sum(scores) / len(scores),
        "Seitwärts: Endpunkte weichen " + ", ".join(fmt_pct(d) for d in devs) + " voneinander ab",
    )


# ---------------------------------------------------------------------------
# general
# ---------------------------------------------------------------------------


def g_proportionality(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    if v.k < 2:
        return _na("Zu wenige Wellen")
    p = cfg.guidelines.params
    durs = [v.dur(i) for i in range(1, v.k + 1)]
    lens = [v.L(i) for i in range(1, v.k + 1)]
    time_ratio = max(durs) / min(durs)
    price_ratio = min(lens) / max(lens) if max(lens) > 0 else 0.0
    limit_t, limit_p, power = p["proportion_max_time_ratio"], p["proportion_min_price_ratio"], p["proportion_decay"]
    t_score = 1.0 if time_ratio <= limit_t else (limit_t / time_ratio) ** power
    p_score = 1.0 if price_ratio >= limit_p else (price_ratio / limit_p) ** power
    return GuidelineResult(
        (t_score + p_score) / 2.0,
        f"Dauer max/min = {fmt_ratio(time_ratio)}, kleinste/größte Welle = {fmt_pct(price_ratio)}",
    )


def g_subwave_structure(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    vals = ctx.subwave_values
    if not vals:
        return _na("Subwellen nicht validiert (keine feineren Pivots)")
    known = [x for x in vals if x is not None]
    if not known:
        return _na("Subwellen nicht validiert (keine feineren Pivots)")
    ok = sum(1 for x in known if x >= 0.5)
    return GuidelineResult(
        sum(known) / len(known),
        f"{ok} von {len(known)} prüfbaren Wellen mit passender Binnenstruktur ({len(vals) - len(known)} nicht validiert)",
    )


def g_completeness(v: CountView, ctx: GuideContext, cfg: Settings) -> GuidelineResult:
    return GuidelineResult(v.k / v.n, f"{v.k} von {v.n} Wellen erkennbar")


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

IMPULSE_GUIDES = (
    Guideline("impulse_w2_retrace", "Retracement Welle 2", g_impulse_w2_retrace),
    Guideline("impulse_w4_retrace", "Retracement Welle 4", g_impulse_w4_retrace),
    Guideline("impulse_alternation", "Alternation W2/W4", g_impulse_alternation),
    Guideline("impulse_extension", "Extension", g_impulse_extension),
    Guideline("impulse_w3_ratio", "Verhältnis W3/W1", g_impulse_w3_ratio),
    Guideline("impulse_w5_ratio", "Verhältnis W5", g_impulse_w5_ratio),
    Guideline("impulse_channel", "Channeling", g_impulse_channel),
    Guideline("impulse_volume", "Volumen W3 > W5", g_impulse_volume),
    Guideline("impulse_momentum", "Momentum-Divergenz W3/W5", g_impulse_momentum),
    Guideline("impulse_time_w2_w4", "Zeitproportion W2/W4", g_impulse_time_w2_w4),
)
DIAGONAL_GUIDES = (Guideline("diagonal_ratios", "Diagonal-Verhältnisse", g_diagonal_ratios),)
ZIGZAG_GUIDES = (
    Guideline("zigzag_b_retrace", "Retracement B", g_zigzag_b_retrace),
    Guideline("zigzag_c_ratio", "Verhältnis C/A", g_zigzag_c_ratio),
)
DOUBLE_ZIGZAG_GUIDES = (Guideline("double_zigzag_y_ratio", "Verhältnis Y/W", g_double_zigzag_y_ratio),)
FLAT_GUIDES = (
    Guideline("flat_b_ratio", "Verhältnis B/A", g_flat_b_ratio),
    Guideline("flat_c_ratio", "Verhältnis C/A", g_flat_c_ratio),
)
TRIANGLE_GUIDES = (Guideline("triangle_ratios", "Dreiecks-Verhältnisse", g_triangle_ratios),)
COMBINATION_GUIDES = (Guideline("combination_sideways", "Seitwärtsverlauf", g_combination_sideways),)
GENERAL_GUIDES = (
    Guideline("proportionality", "Proportionalität", g_proportionality),
    Guideline("subwave_structure", "Subwellen-Struktur", g_subwave_structure),
    Guideline("completeness", "Vollständigkeit", g_completeness),
)
