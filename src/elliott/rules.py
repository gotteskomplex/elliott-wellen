"""Hard Elliott rules (exclusion criteria).

Every rule is a single, individually testable function
``rule(view, cfg) -> RuleResult``. A :class:`RuleResult` unpacks like the tuple
``(passed: bool, reason: str)``; it additionally carries a ``deferred`` flag.

Rules work on a normalized :class:`~elliott.measure.CountView` in which the first
wave always rises (see :mod:`elliott.measure`), so each rule is written once for
up- and down-trending patterns.

Partial counts
    Rules are evaluated incrementally while the search adds waves. A rule that
    needs waves which are not there yet passes with an explanatory reason.

Open last wave
    The last wave of a count that ends at the provisional right-edge pivot may
    still extend. A condition that the wave *can still reach* by extending
    ("wave 3 must exceed the end of wave 1") is then *deferred* (passes, flagged).
    A condition that extension can only make worse ("wave 4 must not enter
    wave 1") is checked immediately, because it already fails for good.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from elliott.fmt import fmt_pct, fmt_price, fmt_ratio
from elliott.measure import CountView
from elliott.models import PATTERN_LABELS, PatternType
from elliott.settings import RulesConfig


@dataclass(frozen=True, slots=True)
class RuleResult:
    """Outcome of a rule check; iterable as ``(passed, reason)``."""

    passed: bool
    reason: str
    deferred: bool = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        yield self.passed
        yield self.reason


RuleFn = Callable[[CountView, RulesConfig], RuleResult]


@dataclass(frozen=True, slots=True)
class Rule:
    """A named hard rule."""

    key: str
    title: str
    func: RuleFn

    def __call__(self, view: CountView, cfg: RulesConfig) -> RuleResult:
        return self.func(view, cfg)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _ok(reason: str) -> RuleResult:
    return RuleResult(True, reason)


def _fail(reason: str) -> RuleResult:
    return RuleResult(False, reason)


def _deferred(reason: str) -> RuleResult:
    return RuleResult(True, reason, deferred=True)


def _lab(v: CountView, wave: int) -> str:
    """Label of wave ``wave`` (1-based), e.g. '3' or 'C'."""
    return PATTERN_LABELS[v.pattern][wave - 1]


def _w(v: CountView, wave: int) -> str:
    return f"Welle {_lab(v, wave)}"


def _p(v: CountView, point: int) -> str:
    return fmt_price(v.raw[point])


def _beyond_word(v: CountView, normalized_down: bool) -> str:
    """German verb for moving beyond a level ('unterschreitet' / 'überschreitet')."""
    price_down = normalized_down == (v.d > 0)
    return "unterschreitet" if price_down else "überschreitet"


def _need(v: CountView, wave: int) -> RuleResult | None:
    if not v.has(wave):
        return _ok(f"{_w(v, wave) if wave <= v.n else 'Welle'} noch nicht vorhanden – nicht prüfbar")
    return None


def _endpoint_can_reach(v: CountView, wave: int, want_higher: bool) -> bool:
    """True if extending open ``wave`` moves its end in the wanted direction."""
    return v.is_open(wave) and (v.wave_up(wave) == want_higher)


# ---------------------------------------------------------------------------
# common
# ---------------------------------------------------------------------------


def rule_alternation(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Waves strictly alternate direction (odd waves with, even against the pattern)."""
    for i in range(1, v.k + 1):
        diff = v.x[i] - v.x[i - 1]
        if abs(diff) <= cfg.equality_tol or (diff > 0) != v.wave_up(i):
            return _fail(f"{_w(v, i)} läuft nicht in die erwartete Richtung")
    return _ok("Wellen wechseln korrekt die Richtung")


def rule_subwave_types(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Subwave patterns must be allowed at their position (context rules).

    Examples: a triangle never appears as wave 2; a leading diagonal only as
    wave 1 or A; an ending diagonal only as wave 5 or C.
    """
    if v.sub_patterns is None or all(s is None for s in v.sub_patterns):
        return _ok("Subwellen-Muster unbekannt – Kontext nicht prüfbar")
    from elliott.patterns import get_spec  # local import: patterns depend on rules

    spec = get_spec(v.pattern)
    for i, sub in enumerate(v.sub_patterns[: v.k]):
        if sub is not None and sub not in spec.sub_types[i]:
            return _fail(f"{_w(v, i + 1)} als {sub.german} ist an dieser Position nicht erlaubt")
    return _ok("Subwellen-Muster passen zu ihren Positionen")


# ---------------------------------------------------------------------------
# impulse (and shared by diagonals)
# ---------------------------------------------------------------------------


def rule_w2_max_retrace(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Wave 2 never retraces more than 100 % of wave 1."""
    if (res := _need(v, 2)) is not None:
        return res
    r = v.L(2) / v.L(1)
    if v.x[2] < v.x[0] - cfg.equality_tol:
        return _fail(
            f"Welle 2 retraced {fmt_pct(r)} von Welle 1 und {_beyond_word(v, True)} den Start von Welle 1 ({_p(v, 0)})"
        )
    return _ok(f"Welle 2 retraced {fmt_pct(r)} von Welle 1 (≤ 100 %)")


def rule_w3_beyond_w1(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Wave 3 must move beyond the end of wave 1."""
    if (res := _need(v, 3)) is not None:
        return res
    if v.x[3] > v.x[1] + cfg.equality_tol:
        return _ok(f"Welle 3 geht über das Ende von Welle 1 ({_p(v, 1)}) hinaus")
    if _endpoint_can_reach(v, 3, want_higher=True):
        return _deferred(f"Welle 3 läuft noch und muss das Ende von Welle 1 ({_p(v, 1)}) noch übertreffen")
    return _fail(f"Welle 3 endet bei {_p(v, 3)} und übertrifft das Ende von Welle 1 ({_p(v, 1)}) nicht")


def rule_w3_not_shortest(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Wave 3 is never the shortest of waves 1, 3 and 5 (in price)."""
    if (res := _need(v, 5)) is not None:
        if v.has(3) and v.L(3) < v.L(1):
            return _ok(
                f"Welle 3 ({fmt_ratio(v.L(3) / v.L(1))} × W1) ist kürzer als Welle 1 – Welle 5 darf Welle 3 nicht übertreffen"
            )
        return res
    l1, l3, l5 = v.L(1), v.L(3), v.L(5)
    tol = cfg.equality_tol
    if l3 < l1 - tol and l3 < l5 - tol:
        return _fail(
            f"Welle 3 ist die kürzeste (W1 = 1, W3 = {fmt_ratio(l3 / l1)}, W5 = {fmt_ratio(l5 / l1)})"
        )
    return _ok(f"Welle 3 ist nicht die kürzeste (W3/W1 = {fmt_ratio(l3 / l1)}, W5/W1 = {fmt_ratio(l5 / l1)})")


def rule_w4_no_overlap(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Wave 4 does not enter the price territory of wave 1."""
    if (res := _need(v, 4)) is not None:
        return res
    if v.x[4] < v.x[1] - cfg.equality_tol:
        return _fail(f"Welle 4 ({_p(v, 4)}) dringt in den Bereich von Welle 1 ein (Ende W1: {_p(v, 1)})")
    return _ok(f"Keine Überlappung: Welle 4 ({_p(v, 4)}) bleibt jenseits des Endes von Welle 1 ({_p(v, 1)})")


# ---------------------------------------------------------------------------
# diagonals
# ---------------------------------------------------------------------------


def rule_diagonal_w4_max_retrace(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Wave 4 of a diagonal does not retrace beyond the start of wave 3."""
    if (res := _need(v, 4)) is not None:
        return res
    if v.x[4] < v.x[2] - cfg.equality_tol:
        return _fail(f"Welle 4 {_beyond_word(v, True)} den Start von Welle 3 ({_p(v, 2)})")
    return _ok(f"Welle 4 retraced {fmt_pct(v.L(4) / v.L(3))} von Welle 3 (≤ 100 %)")


def rule_diagonal_w4_overlap(v: CountView, cfg: RulesConfig) -> RuleResult:
    """In a diagonal wave 4 overlaps wave 1 (otherwise it is an impulse)."""
    if not cfg.diagonal_require_overlap:
        return _ok("Überlappung nicht gefordert (Konfiguration)")
    if (res := _need(v, 4)) is not None:
        return res
    if v.x[4] < v.x[1] - cfg.equality_tol:
        return _ok(f"Welle 4 ({_p(v, 4)}) überlappt mit Welle 1 (Ende W1: {_p(v, 1)})")
    if _endpoint_can_reach(v, 4, want_higher=False):
        return _deferred("Welle 4 läuft noch; für ein Diagonal muss sie in Welle 1 eindringen")
    return _fail("Welle 4 überlappt Welle 1 nicht – das wäre ein Impuls, kein Diagonal")


def _length_conditions(
    v: CountView, pairs: Sequence[tuple[int, int]], longer: bool
) -> tuple[bool, bool]:
    """Check ``L(a) < L(b)`` (or ``>`` if ``longer``) for all present pairs.

    Returns ``(feasible, deferred)``. A ``>`` condition on the open wave that is
    not yet met is deferred (the wave can still grow); a ``<`` condition that is
    violated can only get worse.
    """
    deferred = False
    for a, b in pairs:
        if not v.has(a):
            continue
        la, lb = v.L(a), v.L(b)
        holds = la > lb if longer else la < lb
        if holds:
            continue
        if longer and v.is_open(a):
            deferred = True
            continue
        return False, False
    return True, deferred


_DIAG_PAIRS = ((3, 1), (4, 2), (5, 3))


def rule_diagonal_shape(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Contracting: W3 < W1, W5 < W3, W4 < W2. Expanding: the reverse."""
    if (res := _need(v, 3)) is not None:
        return res
    ok_c, _ = _length_conditions(v, _DIAG_PAIRS, longer=False)
    if ok_c:
        return _ok("Kontrahierendes Diagonal: jede Welle kürzer als die vorherige gleicher Richtung")
    ok_e, deferred = _length_conditions(v, _DIAG_PAIRS, longer=True)
    if ok_e:
        if deferred:
            return _deferred("Expandierendes Diagonal möglich – die laufende Welle muss noch länger werden")
        return _ok("Expandierendes Diagonal: jede Welle länger als die vorherige gleicher Richtung")
    return _fail("Wellenlängen weder konsequent kontrahierend noch expandierend")


def _line_value(t0: int, x0: float, t1: int, x1: float, t: int) -> float:
    if t1 == t0:
        return x1
    return x0 + (x1 - x0) * (t - t0) / (t1 - t0)


def _gap(v: CountView, at: int) -> float:
    """Distance between the 1-3 line and the 2-4 line at bar ``at``."""
    upper = _line_value(v.t[1], v.x[1], v.t[3], v.x[3], at)
    lower = _line_value(v.t[2], v.x[2], v.t[4], v.x[4], at)
    return upper - lower


def rule_diagonal_trendlines(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Trendlines 1-3 and 2-4 converge (contracting) or diverge (expanding)."""
    if (res := _need(v, 4)) is not None:
        return res
    if v.is_open(4):
        return _deferred("Welle 4 läuft noch – Trendlinien 1-3/2-4 noch nicht final")
    g_start, g_end = _gap(v, v.t[1]), _gap(v, v.t[4])
    tol = cfg.diagonal_trendline_tol * v.L(1)
    contracting = v.L(3) < v.L(1)
    if contracting:
        if g_end < g_start - tol:
            return _ok("Trendlinien 1-3 und 2-4 konvergieren")
        return _fail("Kontrahierendes Diagonal, aber die Trendlinien 1-3 und 2-4 konvergieren nicht")
    if g_end > g_start + tol:
        return _ok("Trendlinien 1-3 und 2-4 divergieren")
    return _fail("Expandierendes Diagonal, aber die Trendlinien 1-3 und 2-4 divergieren nicht")


# ---------------------------------------------------------------------------
# zigzags
# ---------------------------------------------------------------------------


def rule_zigzag_b_max_retrace(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Zigzag: B does not retrace beyond the start of A."""
    if (res := _need(v, 2)) is not None:
        return res
    if v.x[2] < v.x[0] - cfg.equality_tol:
        return _fail(f"{_w(v, 2)} {_beyond_word(v, True)} den Start von {_w(v, 1)} ({_p(v, 0)})")
    return _ok(f"{_w(v, 2)} retraced {fmt_pct(v.L(2) / v.L(1))} von {_w(v, 1)} (< 100 %)")


def rule_zigzag_c_beyond_a(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Zigzag: C moves beyond the end of A."""
    if (res := _need(v, 3)) is not None:
        return res
    if v.x[3] > v.x[1] + cfg.equality_tol:
        return _ok(f"{_w(v, 3)} geht über das Ende von {_w(v, 1)} ({_p(v, 1)}) hinaus")
    if _endpoint_can_reach(v, 3, want_higher=True):
        return _deferred(f"{_w(v, 3)} läuft noch und muss das Ende von {_w(v, 1)} ({_p(v, 1)}) noch übertreffen")
    return _fail(f"{_w(v, 3)} übertrifft das Ende von {_w(v, 1)} ({_p(v, 1)}) nicht")


def rule_tz_x2_max_retrace(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Triple zigzag: the second X does not retrace beyond the start of Y."""
    if (res := _need(v, 4)) is not None:
        return res
    if v.x[4] < v.x[2] - cfg.equality_tol:
        return _fail(f"Zweite X-Welle {_beyond_word(v, True)} den Start von Welle Y ({_p(v, 2)})")
    return _ok("Zweite X-Welle bleibt innerhalb von Welle Y")


def rule_tz_z_beyond_y(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Triple zigzag: Z moves beyond the end of Y."""
    if (res := _need(v, 5)) is not None:
        return res
    if v.x[5] > v.x[3] + cfg.equality_tol:
        return _ok(f"Welle Z geht über das Ende von Welle Y ({_p(v, 3)}) hinaus")
    if _endpoint_can_reach(v, 5, want_higher=True):
        return _deferred(f"Welle Z läuft noch und muss das Ende von Welle Y ({_p(v, 3)}) noch übertreffen")
    return _fail(f"Welle Z übertrifft das Ende von Welle Y ({_p(v, 3)}) nicht")


# ---------------------------------------------------------------------------
# flats
# ---------------------------------------------------------------------------


def rule_flat_b_min_retrace(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Flat: B retraces at least ~90 % of A."""
    if (res := _need(v, 2)) is not None:
        return res
    r = v.L(2) / v.L(1)
    if r >= cfg.flat_b_min_retrace - cfg.equality_tol:
        return _ok(f"B retraced {fmt_pct(r)} von A (≥ {fmt_pct(cfg.flat_b_min_retrace)})")
    if v.is_open(2):
        return _deferred(f"B läuft noch ({fmt_pct(r)} von A) – für ein Flat sind ≥ {fmt_pct(cfg.flat_b_min_retrace)} nötig")
    return _fail(f"B retraced nur {fmt_pct(r)} von A (< {fmt_pct(cfg.flat_b_min_retrace)})")


def rule_flat_b_max_retrace(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Flat: plausibility cap for B (configurable)."""
    if (res := _need(v, 2)) is not None:
        return res
    r = v.L(2) / v.L(1)
    if r > cfg.flat_b_max_retrace:
        return _fail(f"B ist mit {fmt_pct(r)} von A zu lang für ein Flat (max. {fmt_pct(cfg.flat_b_max_retrace)})")
    return _ok(f"B = {fmt_pct(r)} von A (≤ {fmt_pct(cfg.flat_b_max_retrace)})")


def rule_flat_regular_b(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Regular flat: B ends about at the start of A (≤ configured maximum)."""
    if (res := _need(v, 2)) is not None:
        return res
    r = v.L(2) / v.L(1)
    if r > cfg.flat_regular_b_max:
        return _fail(f"B = {fmt_pct(r)} von A – zu lang für ein Regular Flat (≤ {fmt_pct(cfg.flat_regular_b_max)})")
    return _ok(f"B = {fmt_pct(r)} von A (Regular Flat)")


def rule_flat_regular_c(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Regular flat: C ends about at or slightly beyond the end of A."""
    if (res := _need(v, 3)) is not None:
        return res
    bound = v.x[1] - cfg.flat_regular_c_shortfall * v.L(1)
    if v.x[3] >= bound - cfg.equality_tol:
        return _ok(f"C erreicht das Ende von A ({_p(v, 1)}) bzw. liegt knapp dahinter")
    if _endpoint_can_reach(v, 3, want_higher=True):
        return _deferred(f"C läuft noch und muss das Ende von A ({_p(v, 1)}) noch erreichen")
    return _fail(f"C verfehlt das Ende von A ({_p(v, 1)}) deutlich")


def rule_flat_irregular_b(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Expanded/running flat: B moves beyond the start of A (> configured minimum)."""
    if (res := _need(v, 2)) is not None:
        return res
    r = v.L(2) / v.L(1)
    if r > cfg.flat_expanded_b_min:
        return _ok(f"B = {fmt_pct(r)} von A – jenseits des A-Starts ({_p(v, 0)})")
    if v.is_open(2):
        return _deferred(f"B läuft noch ({fmt_pct(r)} von A) und muss den Start von A ({_p(v, 0)}) übertreffen")
    return _fail(f"B = {fmt_pct(r)} von A – übertrifft den Start von A nicht deutlich (> {fmt_pct(cfg.flat_expanded_b_min)})")


def rule_flat_expanded_c(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Expanded flat: C moves beyond the end of A."""
    if (res := _need(v, 3)) is not None:
        return res
    if v.x[3] > v.x[1] + cfg.equality_tol:
        return _ok(f"C geht über das Ende von A ({_p(v, 1)}) hinaus")
    if _endpoint_can_reach(v, 3, want_higher=True):
        return _deferred(f"C läuft noch und muss das Ende von A ({_p(v, 1)}) noch übertreffen")
    return _fail(f"C erreicht das Ende von A ({_p(v, 1)}) nicht (das wäre ein Running Flat)")


def rule_flat_running_c(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Running flat: C does not reach the end of A, but is not negligible."""
    if (res := _need(v, 3)) is not None:
        return res
    if v.x[3] > v.x[1] + cfg.equality_tol:
        return _fail(f"C übertrifft das Ende von A ({_p(v, 1)}) – kein Running Flat")
    r = v.L(3) / v.L(1)
    if r >= cfg.flat_running_c_min:
        return _ok(f"C = {fmt_pct(r)} von A und bleibt vor dem Ende von A ({_p(v, 1)})")
    if v.is_open(3):
        return _deferred(f"C läuft noch ({fmt_pct(r)} von A)")
    return _fail(f"C ist mit {fmt_pct(r)} von A zu kurz (min. {fmt_pct(cfg.flat_running_c_min)})")


# ---------------------------------------------------------------------------
# triangles
# ---------------------------------------------------------------------------

_TRI_PAIRS = ((3, 1), (4, 2), (5, 3))


def rule_triangle_contracting(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Contracting triangle: each wave shorter than the previous one of the same
    direction; line A–C falls, line B–D rises (in pattern direction)."""
    if (res := _need(v, 3)) is not None:
        return res
    ok, _ = _length_conditions(v, _TRI_PAIRS, longer=False)
    if not ok:
        return _fail("Nicht jede Welle ist kürzer als die vorherige gleicher Richtung")
    tol = cfg.triangle_line_tol * v.L(1)
    if not v.x[3] < v.x[1] - tol:
        return _fail("Linie A–C verläuft nicht konvergierend (C endet nicht innerhalb von A)")
    if v.has(4) and not v.x[4] > v.x[2] + tol:
        return _fail("Linie B–D verläuft nicht konvergierend")
    return _ok("Kontrahierend: Wellen werden kürzer, Linien A–C und B–D konvergieren")


def rule_triangle_barrier(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Barrier triangle: contracting, but one boundary line is ~horizontal."""
    if (res := _need(v, 3)) is not None:
        return res
    for a, b in ((3, 1), (5, 3)):
        if v.has(a) and not v.L(a) < v.L(b):
            return _fail(f"{_w(v, a)} ist nicht kürzer als {_w(v, b)}")
    tol = cfg.triangle_line_tol * v.L(1)
    if v.has(4) and v.L(4) > v.L(2) + tol:
        return _fail("D ist deutlich länger als B")
    ac_flat = abs(v.x[3] - v.x[1]) <= tol
    ac_falling = v.x[3] < v.x[1] - tol
    if not (ac_flat or ac_falling):
        return _fail("Linie A–C divergiert")
    if not v.has(4):
        if ac_flat:
            return _ok("Linie A–C horizontal (Barriere)")
        return _deferred("Linie A–C fällt – Linie B–D muss horizontal werden")
    bd_flat = abs(v.x[4] - v.x[2]) <= tol
    bd_rising = v.x[4] > v.x[2] + tol
    if ac_flat and bd_rising:
        return _ok("Barriere A–C horizontal, Linie B–D steigt")
    if bd_flat and ac_falling:
        return _ok("Barriere B–D horizontal, Linie A–C fällt")
    if ac_falling and bd_rising and v.is_open(4):
        return _deferred("D läuft noch – Linie B–D muss horizontal werden")
    return _fail("Keine der Begrenzungslinien ist horizontal (bzw. beide)")


def rule_triangle_expanding(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Expanding triangle: each wave longer than the previous one of the same
    direction; lines A–C and B–D diverge."""
    if (res := _need(v, 3)) is not None:
        return res
    ok, deferred = _length_conditions(v, _TRI_PAIRS, longer=True)
    if not ok:
        return _fail("Nicht jede Welle ist länger als die vorherige gleicher Richtung")
    tol = cfg.triangle_line_tol * v.L(1)
    if not v.x[3] > v.x[1] + tol:
        if not v.is_open(3):
            return _fail("Linie A–C divergiert nicht")
        deferred = True
    if v.has(4) and not v.x[4] < v.x[2] - tol:
        if not v.is_open(4):
            return _fail("Linie B–D divergiert nicht")
        deferred = True
    if deferred:
        return _deferred("Expandierendes Dreieck möglich – laufende Welle muss noch länger werden")
    return _ok("Expandierend: Wellen werden länger, Linien A–C und B–D divergieren")


# ---------------------------------------------------------------------------
# combinations
# ---------------------------------------------------------------------------


def rule_combination_x_max(v: CountView, cfg: RulesConfig) -> RuleResult:
    """X waves stay within a plausible size relative to the preceding wave."""
    for x_wave in (2, 4):
        if v.has(x_wave):
            r = v.L(x_wave) / v.L(x_wave - 1)
            if r > cfg.combination_x_max:
                return _fail(
                    f"X-Welle ist mit {fmt_pct(r)} der vorherigen Welle zu groß (max. {fmt_pct(cfg.combination_x_max)})"
                )
    return _ok("X-Wellen in plausibler Größe")


def rule_combination_sideways(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Combinations move sideways: Y (and Z) do not travel far beyond W."""
    if (res := _need(v, 3)) is not None:
        return res
    limit = cfg.combination_max_overshoot * v.L(1)
    if v.x[3] > v.x[1] + limit:
        return _fail("Welle Y läuft weit über W hinaus – eher Double Zigzag als seitwärts")
    if v.has(5) and v.x[5] > max(v.x[1], v.x[3]) + limit:
        return _fail("Welle Z läuft weit über W/Y hinaus – nicht seitwärts")
    return _ok("Seitwärtsverlauf: Y/Z bleiben in der Nähe des W-Endes")


def rule_combination_triangles(v: CountView, cfg: RulesConfig) -> RuleResult:
    """Combinations contain at most one triangle, and only as the last wave."""
    if v.sub_patterns is None:
        return _ok("Subwellen-Muster unbekannt – nicht prüfbar")
    tri = [i for i, s in enumerate(v.sub_patterns[: v.k]) if s is not None and s.value.startswith("triangle")]
    if len(tri) > cfg.combination_max_triangles:
        return _fail(f"{len(tri)} Dreiecke in der Kombination (max. {cfg.combination_max_triangles})")
    if tri and tri[0] != v.n - 1:
        return _fail("Dreieck nur als letzte Welle einer Kombination erlaubt")
    return _ok("Dreiecks-Platzierung in Ordnung")


# ---------------------------------------------------------------------------
# registry & evaluation
# ---------------------------------------------------------------------------

ALTERNATION = Rule("alternation", "Richtungswechsel", rule_alternation)
SUBWAVE_TYPES = Rule("subwave_types", "Subwellen-Kontext", rule_subwave_types)
W2_MAX = Rule("w2_max_retrace", "Welle 2 < 100 % von Welle 1", rule_w2_max_retrace)
W3_BEYOND_W1 = Rule("w3_beyond_w1", "Welle 3 über Ende Welle 1", rule_w3_beyond_w1)
W3_NOT_SHORTEST = Rule("w3_not_shortest", "Welle 3 nie die kürzeste", rule_w3_not_shortest)
W4_NO_OVERLAP = Rule("w4_no_overlap", "Welle 4 ohne Überlappung mit Welle 1", rule_w4_no_overlap)
DIAG_W4_MAX = Rule("diagonal_w4_max_retrace", "Welle 4 < 100 % von Welle 3", rule_diagonal_w4_max_retrace)
DIAG_W4_OVERLAP = Rule("diagonal_w4_overlap", "Welle 4 überlappt Welle 1", rule_diagonal_w4_overlap)
DIAG_SHAPE = Rule("diagonal_shape", "Diagonal kontrahierend/expandierend", rule_diagonal_shape)
DIAG_LINES = Rule("diagonal_trendlines", "Diagonal-Trendlinien", rule_diagonal_trendlines)
ZZ_B_MAX = Rule("zigzag_b_max_retrace", "B nicht über Start von A", rule_zigzag_b_max_retrace)
ZZ_C_BEYOND = Rule("zigzag_c_beyond_a", "C über Ende von A", rule_zigzag_c_beyond_a)
TZ_X2_MAX = Rule("tz_x2_max_retrace", "Zweite X-Welle < 100 % von Y", rule_tz_x2_max_retrace)
TZ_Z_BEYOND = Rule("tz_z_beyond_y", "Z über Ende von Y", rule_tz_z_beyond_y)
FLAT_B_MIN = Rule("flat_b_min_retrace", "Flat: B ≥ ~90 % von A", rule_flat_b_min_retrace)
FLAT_B_MAX = Rule("flat_b_max_retrace", "Flat: B-Obergrenze", rule_flat_b_max_retrace)
FLAT_REG_B = Rule("flat_regular_b", "Regular Flat: B ≈ 100 %", rule_flat_regular_b)
FLAT_REG_C = Rule("flat_regular_c", "Regular Flat: C erreicht A-Ende", rule_flat_regular_c)
FLAT_IRR_B = Rule("flat_irregular_b", "Expanded/Running: B > 100 %", rule_flat_irregular_b)
FLAT_EXP_C = Rule("flat_expanded_c", "Expanded Flat: C über A-Ende", rule_flat_expanded_c)
FLAT_RUN_C = Rule("flat_running_c", "Running Flat: C vor A-Ende", rule_flat_running_c)
TRI_CONTRACTING = Rule("triangle_contracting", "Dreieck kontrahierend", rule_triangle_contracting)
TRI_BARRIER = Rule("triangle_barrier", "Barrier-Dreieck", rule_triangle_barrier)
TRI_EXPANDING = Rule("triangle_expanding", "Dreieck expandierend", rule_triangle_expanding)
COMB_X_MAX = Rule("combination_x_max", "X-Wellen-Größe", rule_combination_x_max)
COMB_SIDEWAYS = Rule("combination_sideways", "Kombination seitwärts", rule_combination_sideways)
COMB_TRIANGLES = Rule("combination_triangles", "Max. ein Dreieck, nur am Ende", rule_combination_triangles)

ALL_RULES: dict[str, Rule] = {
    r.key: r
    for r in (
        ALTERNATION, SUBWAVE_TYPES, W2_MAX, W3_BEYOND_W1, W3_NOT_SHORTEST, W4_NO_OVERLAP,
        DIAG_W4_MAX, DIAG_W4_OVERLAP, DIAG_SHAPE, DIAG_LINES, ZZ_B_MAX, ZZ_C_BEYOND,
        TZ_X2_MAX, TZ_Z_BEYOND, FLAT_B_MIN, FLAT_B_MAX, FLAT_REG_B, FLAT_REG_C, FLAT_IRR_B,
        FLAT_EXP_C, FLAT_RUN_C, TRI_CONTRACTING, TRI_BARRIER, TRI_EXPANDING, COMB_X_MAX,
        COMB_SIDEWAYS, COMB_TRIANGLES,
    )
}


def evaluate_rules(rules: Iterable[Rule], view: CountView, cfg: RulesConfig) -> list[tuple[Rule, RuleResult]]:
    """Evaluate all rules (no short-circuit) – used for reports and re-checks."""
    return [(rule, rule(view, cfg)) for rule in rules]


def first_failure(rules: Iterable[Rule], view: CountView, cfg: RulesConfig) -> tuple[Rule, RuleResult] | None:
    """Return the first failing rule or ``None`` (short-circuit, used by the search)."""
    for rule in rules:
        res = rule(view, cfg)
        if not res.passed:
            return rule, res
    return None


def passes(rules: Iterable[Rule], view: CountView, cfg: RulesConfig) -> bool:
    """True if no rule fails."""
    return first_failure(rules, view, cfg) is None


def pattern_labels(pattern: PatternType) -> tuple[str, ...]:
    return PATTERN_LABELS[pattern]
