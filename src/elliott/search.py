"""Search for rule-conforming wave counts (beam search with early pruning).

Overview
--------
* The top-level search runs on one pivot degree (the *search level*). From each
  start candidate it builds counts wave by wave for every pattern type. The
  last wave must end at the provisional right-edge pivot, so every count
  describes the *present*: complete patterns ("Muster evtl. abgeschlossen") as
  well as incomplete ones ("wir sind in Welle 3 von 5").
* Pivots may be skipped – they then belong to a subwave. A wave from pivot
  ``p`` to pivot ``j`` is only allowed if ``j`` is the extreme of the segment
  (and, for motive waves, ``p`` as well). Corrective waves may have an
  irregular start (e.g. the B wave of an expanded flat exceeding the start),
  limited by ``irregular_start_max_excess``.
* All hard rules are checked every time a wave is added; a violation prunes
  the branch immediately. Beam width, maximum skip and a time budget bound the
  search.
* Subwave validation checks recursively (depth configurable) whether a wave's
  inner pivots on finer degrees admit a count of an allowed type (e.g. a
  5-wave impulse for wave 3). Without finer pivots the wave is marked
  "nicht validiert" and scored neutrally.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

from elliott.guidelines import GuideContext
from elliott.measure import CountView, PriceScale, make_view
from elliott.models import PatternType, Pivot, SubwaveStatus
from elliott.patterns import SPECS, PatternSpec
from elliott.pivots.zigzag import PivotHierarchy
from elliott.rules import first_failure, has_deferred
from elliott.scoring import quick_score, score_view
from elliott.models import ScoreBreakdown
from elliott.settings import Settings

TOP_LEVEL_TYPES: tuple[PatternType, ...] = tuple(SPECS)


# ---------------------------------------------------------------------------
# pivot arrays
# ---------------------------------------------------------------------------


@dataclass
class PivotArrays:
    """Numpy view of one pivot degree plus segment extremes for fast checks."""

    level: int
    pivots: list[Pivot]
    bars: np.ndarray
    price: np.ndarray
    tp: np.ndarray
    kind: np.ndarray
    max_between: np.ndarray
    min_between: np.ndarray
    suffix_max: np.ndarray
    suffix_min: np.ndarray
    pos_of_bar: dict[tuple[int, int], int]

    @classmethod
    def build(cls, level: int, pivots: list[Pivot], scale: PriceScale) -> "PivotArrays":
        n = len(pivots)
        bars = np.array([p.idx for p in pivots], dtype=int)
        price = np.array([p.price for p in pivots], dtype=float)
        tp = scale.tr_array(price) if n else np.zeros(0)
        kind = np.array([p.kind.sign for p in pivots], dtype=int)
        max_b = np.full((n, n), -np.inf)
        min_b = np.full((n, n), np.inf)
        for i in range(n - 2):
            seg = tp[i + 1 :]
            max_b[i, i + 2 :] = np.maximum.accumulate(seg)[:-1]
            min_b[i, i + 2 :] = np.minimum.accumulate(seg)[:-1]
        suf_max = np.maximum.accumulate(tp[::-1])[::-1] if n else np.zeros(0)
        suf_min = np.minimum.accumulate(tp[::-1])[::-1] if n else np.zeros(0)
        pos = {(int(b), int(k)): i for i, (b, k) in enumerate(zip(bars, kind))}
        return cls(level, pivots, bars, price, tp, kind, max_b, min_b, suf_max, suf_min, pos)

    def __len__(self) -> int:
        return len(self.pivots)

    @property
    def last(self) -> int:
        return len(self.pivots) - 1


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class SubwaveInfo:
    """Result of validating the internal structure of one wave."""

    status: SubwaveStatus
    value: float | None
    note: str
    candidate: "Candidate | None" = None

    @property
    def pattern(self) -> PatternType | None:
        return self.candidate.spec.type if self.candidate is not None else None


@dataclass
class Candidate:
    """A count found by the search."""

    spec: PatternSpec
    level: int
    positions: tuple[int, ...]
    view: CountView
    prelim: float
    pullback: bool = False
    subwaves: list[SubwaveInfo] | None = None
    score: ScoreBreakdown | None = None
    start_level: int = 0

    @property
    def k(self) -> int:
        return len(self.positions) - 1

    @property
    def total(self) -> float:
        return self.score.total if self.score is not None else self.prelim


@dataclass
class SearchStats:
    states: int = 0
    budget_exhausted: bool = False
    started: float = field(default_factory=time.perf_counter)
    deadline: float = float("inf")

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------


class SearchEngine:
    """Enumerates counts over a :class:`PivotHierarchy`."""

    def __init__(
        self,
        hierarchy: PivotHierarchy,
        scale: PriceScale,
        cfg: Settings,
        ctx: GuideContext | None = None,
    ) -> None:
        self.h = hierarchy
        self.scale = scale
        self.cfg = cfg
        self.ctx = ctx or GuideContext()
        self.arrays = [PivotArrays.build(lvl, piv, scale) for lvl, piv in enumerate(hierarchy.levels)]
        self.stats = SearchStats()
        self._sub_cache: dict[tuple[int, int, int, frozenset[PatternType], int], SubwaveInfo] = {}
        self._prior_cache: dict[tuple[int, int], float | None] = {}
        self.top_ctx = self.ctx

    def prior_move(self, level: int, pos: int) -> float | None:
        """Length of the opposite move that ended at pivot ``pos`` (finest degree).

        Walks back from the pivot until price exceeds it again; the extreme on
        the way is the start of the move that ``pos`` terminates.
        """
        key = (level, pos)
        if key in self._prior_cache:
            return self._prior_cache[key]
        arr = self.arrays[level]
        fine = self.arrays[-1]
        fpos = fine.pos_of_bar.get((int(arr.bars[pos]), int(arr.kind[pos])))
        result: float | None = None
        if fpos is not None and fpos > 0:
            base = fine.tp[fpos]
            sign = int(fine.kind[fpos])  # +1 high: prior move was up (from a low)
            extreme = base
            for j in range(fpos - 1, -1, -1):
                v = fine.tp[j]
                if sign < 0:
                    if fine.kind[j] < 0 and v < base:
                        break
                    extreme = max(extreme, v)
                else:
                    if fine.kind[j] > 0 and v > base:
                        break
                    extreme = min(extreme, v)
            result = abs(extreme - base) or None
        self._prior_cache[key] = result
        return result

    # -- configuration helpers ---------------------------------------------

    def choose_level(self) -> int:
        """Search level: configured level, adjusted by pivot count limits."""
        sc = self.cfg.search
        lvl = min(sc.level, len(self.arrays) - 1)
        while lvl > 0 and len(self.arrays[lvl]) > sc.max_pivots:
            lvl -= 1
        while lvl < len(self.arrays) - 1 and len(self.arrays[lvl]) < sc.min_pivots:
            lvl += 1
        return lvl

    def start_candidates(self, level: int, start_bar: int | None = None) -> list[int]:
        """Start positions for the top-level search."""
        arr = self.arrays[level]
        sc = self.cfg.search
        if len(arr) < 2:
            return []
        max_start = arr.last - 1
        if start_bar is not None:
            pos = int(np.argmin(np.abs(arr.bars - start_bar)))
            return [min(pos, max_start)]
        lo = max(0, arr.last - sc.max_start_lookback)
        cands: list[int] = []
        # Significant turning points: pivots of the coarsest degrees.
        for coarse in self.arrays[: level + 1]:
            for p in coarse.pivots:
                pos = arr.pos_of_bar.get((p.idx, p.kind.sign))
                if pos is not None and lo <= pos <= max_start and pos not in cands:
                    cands.append(pos)
        window = arr.tp[lo : max_start + 1]
        if len(window):
            for pos in (lo + int(np.argmax(window)), lo + int(np.argmin(window)), lo):
                if pos not in cands:
                    cands.append(pos)
        # Prefer the most significant (largest swing to the right edge), keep limit.
        cands.sort(key=lambda p: abs(arr.tp[p] - arr.tp[arr.last]), reverse=True)
        return sorted(cands[: sc.max_start_candidates])

    # -- core enumeration ----------------------------------------------------

    def _clean(self, arr: PivotArrays, p: int, j: int, up: bool, relaxed: bool) -> tuple[bool, bool]:
        """Segment check for a wave ``p -> j``.

        Returns ``(valid, stop)``; ``stop`` means no later ``j`` can be valid
        (the start was exceeded without an allowance).
        """
        tp = arr.tp
        tol = self.cfg.rules.equality_tol
        if up:
            if tp[j] <= tp[p]:
                return False, False
            if arr.max_between[p, j] > tp[j] + tol:
                return False, False
            allowance = self.cfg.search.irregular_start_max_excess * (tp[j] - tp[p]) if relaxed else 0.0
            if arr.min_between[p, j] < tp[p] - allowance - tol:
                return False, not relaxed
            return True, False
        if tp[j] >= tp[p]:
            return False, False
        if arr.min_between[p, j] < tp[j] - tol:
            return False, False
        allowance = self.cfg.search.irregular_start_max_excess * (tp[p] - tp[j]) if relaxed else 0.0
        if arr.max_between[p, j] > tp[p] + allowance + tol:
            return False, not relaxed
        return True, False

    def _running_inside(self, arr: PivotArrays, p: int, j: int, up: bool, relaxed: bool) -> bool:
        """True if every pivot after ``j`` stays inside the wave ``p -> j``.

        The wave can then still be running: ``j`` is its extreme so far and the
        later pivots are an internal pullback (or the start of its last subwave).
        """
        if j >= arr.last:
            return False
        tol = self.cfg.rules.equality_tol
        length = abs(arr.tp[j] - arr.tp[p])
        allowance = self.cfg.search.irregular_start_max_excess * length if relaxed else 0.0
        hi, lo = arr.suffix_max[j + 1], arr.suffix_min[j + 1]
        if up:
            return bool(hi <= arr.tp[j] + tol and lo > arr.tp[p] - allowance + tol)
        return bool(lo >= arr.tp[j] - tol and hi < arr.tp[p] + allowance - tol)

    def make_view(
        self,
        spec: PatternSpec,
        arr: PivotArrays,
        positions: Sequence[int],
        last_open: bool,
        sub_patterns: tuple[PatternType | None, ...] | None = None,
    ) -> CountView:
        d = 1 if arr.kind[positions[0]] < 0 else -1
        idx = list(positions)
        return make_view(
            spec.type, spec.n, d, arr.tp[idx].tolist(), arr.bars[idx].tolist(), arr.price[idx].tolist(),
            last_open, sub_patterns, self.prior_move(arr.level, positions[0]),
        )

    def enumerate(
        self,
        level: int,
        start: int,
        types: Iterable[PatternType],
        end: int | None = None,
        min_waves: int | None = None,
        max_skip: int | None = None,
        beam_width: int | None = None,
        ctx: GuideContext | None = None,
    ) -> list[Candidate]:
        """Enumerate counts starting at ``start``.

        Args:
            level: pivot degree.
            start: start position.
            types: pattern types to try.
            end: fixed end position (complete patterns only). ``None`` = the
                count must end at the right-edge pivot and may be incomplete.
            min_waves: minimum number of waves for incomplete counts.
        """
        arr = self.arrays[level]
        sc = self.cfg.search
        edge = end is None
        terminal = arr.last if edge else end
        max_skip = sc.max_skip if max_skip is None else max_skip
        beam_width = sc.beam_width if beam_width is None else beam_width
        min_waves = sc.min_waves if min_waves is None else min_waves
        step_max = 2 * max_skip + 1
        results: list[Candidate] = []
        if terminal <= start:
            return results
        d = 1 if arr.kind[start] < 0 else -1
        rules_cfg = self.cfg.rules
        ctx = ctx or self.ctx

        for ptype in types:
            spec = SPECS[ptype]
            beam: list[tuple[int, ...]] = [(start,)]
            for k in range(1, spec.n + 1):
                if time.perf_counter() > self.stats.deadline:
                    self.stats.budget_exhausted = True
                    break
                up = (d > 0) == (k % 2 == 1)
                relaxed = spec.relaxed_start(k)
                remaining = spec.n - k
                expanded: list[tuple[tuple[int, ...], CountView]] = []
                for state in beam:
                    p = state[-1]
                    j_hi = min(terminal, p + step_max)
                    for j in range(p + 1, j_hi + 1, 2):
                        if not edge:
                            gap = terminal - j
                            if remaining == 0 and gap != 0:
                                continue
                            if remaining > 0 and (gap < remaining or gap > remaining * step_max):
                                continue
                        valid, stop = self._clean(arr, p, j, up, relaxed)
                        if stop:
                            break
                        if not valid:
                            continue
                        positions = state + (j,)
                        is_terminal = j == terminal
                        # "Running" end: the last wave's extreme so far is j and all later
                        # pivots stay inside the wave (internal pullback).
                        is_pullback = edge and j < terminal and self._running_inside(arr, p, j, up, relaxed)
                        accept = k == spec.n or (edge and k >= min_waves)
                        # can the remaining waves still reach the right edge?
                        reachable = remaining > 0 and terminal - j <= remaining * step_max
                        if edge and not (is_terminal or is_pullback or reachable):
                            continue
                        view = self.make_view(spec, arr, positions, last_open=edge and is_terminal)
                        self.stats.states += 1
                        if first_failure(spec.all_rules, view, rules_cfg) is not None:
                            if not (is_pullback and accept):
                                continue
                            # A target rule may be deferred while the wave still runs.
                            open_view = self.make_view(spec, arr, positions, last_open=True)
                            open_view.running = True
                            if first_failure(spec.all_rules, open_view, rules_cfg) is None:
                                prelim = quick_score(spec, open_view, ctx, self.cfg)
                                results.append(Candidate(spec, level, positions, open_view, prelim, pullback=True))
                            continue
                        if is_terminal:
                            if accept:
                                view.running = edge and has_deferred(spec.all_rules, view, rules_cfg)
                                prelim = quick_score(spec, view, ctx, self.cfg)
                                results.append(Candidate(spec, level, positions, view, prelim))
                            continue
                        if is_pullback and accept:
                            open_view = self.make_view(spec, arr, positions, last_open=True)
                            open_view.running = True
                            if first_failure(spec.all_rules, open_view, rules_cfg) is None:
                                prelim = quick_score(spec, open_view, ctx, self.cfg)
                                results.append(Candidate(spec, level, positions, open_view, prelim, pullback=True))
                        if k < spec.n and (reachable or not edge):
                            expanded.append((positions, view))
                if len(expanded) > beam_width:
                    scored = sorted(
                        expanded, key=lambda sv: quick_score(spec, sv[1], ctx, self.cfg), reverse=True
                    )
                    expanded = scored[:beam_width]
                beam = [s for s, _ in expanded]
                if not beam:
                    break
        return results

    # -- subwave validation -------------------------------------------------

    def _sub_level(self, parent_level: int, bar_a: int, bar_b: int) -> int:
        """Finest degree (>= parent) whose inner pivot count stays tractable."""
        limit = self.cfg.search.subwave_max_inner_pivots
        for lvl in range(len(self.arrays) - 1, parent_level - 1, -1):
            arr = self.arrays[lvl]
            inner = int(np.count_nonzero((arr.bars > bar_a) & (arr.bars < bar_b)))
            if inner <= limit:
                return lvl
        return parent_level

    def validate_wave(
        self,
        parent_level: int,
        start: Pivot | tuple[int, int],
        end: Pivot | tuple[int, int],
        allowed: frozenset[PatternType],
        depth: int,
    ) -> SubwaveInfo:
        """Check whether the inner pivots of a wave admit an allowed count."""
        a = (start.idx, start.kind.sign) if isinstance(start, Pivot) else start
        b = (end.idx, end.kind.sign) if isinstance(end, Pivot) else end
        key = (parent_level, a[0], b[0], allowed, depth)
        if key in self._sub_cache:
            return self._sub_cache[key]
        info = self._validate(parent_level, a, b, allowed, depth)
        self._sub_cache[key] = info
        return info

    def _validate(
        self, parent_level: int, a: tuple[int, int], b: tuple[int, int], allowed: frozenset[PatternType], depth: int
    ) -> SubwaveInfo:
        sc = self.cfg.search
        if depth <= 0:
            return SubwaveInfo(SubwaveStatus.NOT_VALIDATED, None, "Rekursionstiefe erreicht")
        lvl = self._sub_level(parent_level, a[0], b[0])
        arr = self.arrays[lvl]
        s_pos, e_pos = arr.pos_of_bar.get(a), arr.pos_of_bar.get(b)
        if s_pos is None or e_pos is None or e_pos <= s_pos:
            return SubwaveInfo(SubwaveStatus.NOT_VALIDATED, None, "Keine feineren Pivots verfügbar")
        inner = e_pos - s_pos - 1
        five_only = all(SPECS[t].n == 5 for t in allowed)
        min_inner = sc.subwave_min_inner_motive if five_only else sc.subwave_min_inner_corrective
        if inner == 0:
            return SubwaveInfo(SubwaveStatus.NOT_VALIDATED, None, "Keine feineren Pivots innerhalb der Welle")
        if inner < min_inner:
            if five_only:
                return SubwaveInfo(SubwaveStatus.FAILED, 0.0, f"Nur {inner + 1} Teilbewegungen erkennbar – 5 erwartet")
            return SubwaveInfo(SubwaveStatus.NOT_VALIDATED, None, "Zu wenige feinere Pivots")
        cands = self.enumerate(
            lvl, s_pos, sorted(allowed, key=lambda t: t.value), end=e_pos,
            max_skip=sc.subwave_max_skip, beam_width=sc.subwave_beam_width,
        )
        if not cands:
            names = "/".join(sorted({SPECS[t].family for t in allowed}))
            return SubwaveInfo(SubwaveStatus.FAILED, 0.0, f"Keine regelkonforme Binnenstruktur ({names}) gefunden")
        if depth > 1:
            for c in cands:
                self.attach_subwaves(c, depth - 1)
        # Prefer subcounts that explain all inner pivots (fewer skipped pivots).
        def quality(c: Candidate) -> float:
            fit = (c.k + 1) / (inner + 2)
            return c.total / 100.0 * (1.0 - sc.subwave_fit_weight + sc.subwave_fit_weight * fit)

        best = max(cands, key=lambda c: (quality(c), -c.spec.n))
        base = sc.subwave_validated_base
        value = base + (1.0 - base) * quality(best)
        return SubwaveInfo(SubwaveStatus.VALIDATED, value, f"Binnenstruktur: {best.spec.type.german}", best)

    def attach_subwaves(self, cand: Candidate, depth: int, top: bool = False) -> None:
        """Validate every wave of a candidate and compute its full score."""
        arr = self.arrays[cand.level]
        infos: list[SubwaveInfo] = []
        for i in range(1, cand.k + 1):
            a, b = cand.positions[i - 1], cand.positions[i]
            allowed = cand.spec.sub_types[i - 1]
            if cand.view.is_open(i):
                infos.append(SubwaveInfo(SubwaveStatus.NOT_VALIDATED, None, "Welle läuft noch"))
                continue
            infos.append(
                self.validate_wave(
                    cand.level, (int(arr.bars[a]), int(arr.kind[a])), (int(arr.bars[b]), int(arr.kind[b])), allowed, depth
                )
            )
        cand.subwaves = infos
        subs = tuple(info.pattern for info in infos)
        running = cand.view.running
        cand.view = self.make_view(cand.spec, arr, cand.positions, cand.view.last_open, subs)
        cand.view.running = running
        base = self.top_ctx if top else self.ctx
        ctx = GuideContext(
            volume=base.volume, rsi=base.rsi, subwave_values=tuple(info.value for info in infos),
            reference_range=base.reference_range,
        )
        cand.score = score_view(cand.spec, cand.view, ctx, self.cfg)

    def current_subcount(self, cand: Candidate) -> Candidate | None:
        """Best (possibly incomplete) count of the last wave on a finer degree."""
        arr = self.arrays[cand.level]
        a = cand.positions[-2]
        start_key = (int(arr.bars[a]), int(arr.kind[a]))
        for lvl in range(len(self.arrays) - 1, cand.level - 1, -1):
            sub = self.arrays[lvl]
            s_pos = sub.pos_of_bar.get(start_key)
            if s_pos is None or sub.last - s_pos < 2 or sub.last - s_pos > self.cfg.search.subwave_max_inner_pivots:
                continue
            allowed = cand.spec.sub_types[cand.k - 1]
            cands = self.enumerate(
                lvl, s_pos, sorted(allowed, key=lambda t: t.value), end=None, min_waves=1,
                max_skip=self.cfg.search.subwave_max_skip, beam_width=self.cfg.search.subwave_beam_width,
            )
            if cands:
                return max(cands, key=lambda c: (c.total, c.k))
        return None

    # -- top level ------------------------------------------------------------

    def search(self, start_bar: int | None = None, level: int | None = None) -> tuple[int, list[Candidate]]:
        """Run the full top-level search; returns ``(search_level, ranked candidates)``."""
        sc = self.cfg.search
        self.stats = SearchStats()
        self.stats.deadline = self.stats.started + sc.time_budget_s
        lvl = self.choose_level() if level is None else level
        arr = self.arrays[lvl]
        if len(arr) < 3:
            return lvl, []
        starts = self.start_candidates(lvl, start_bar)
        if starts:
            window = arr.tp[min(starts) :]
            self.top_ctx = GuideContext(
                volume=self.ctx.volume, rsi=self.ctx.rsi, reference_range=float(window.max() - window.min()) or None
            )
        found: list[Candidate] = []
        for start in starts:
            for c in self.enumerate(lvl, start, TOP_LEVEL_TYPES, ctx=self.top_ctx):
                c.start_level = lvl
                found.append(c)
            if self.stats.budget_exhausted:
                break
        found = dedupe(found, self.cfg)
        found.sort(key=lambda c: c.prelim, reverse=True)
        found = found[: sc.rescore_top]
        for c in found:
            self.attach_subwaves(c, sc.subwave_depth, top=True)
        found = dedupe(found, self.cfg)
        found.sort(key=lambda c: (-c.total, complexity_key(c, self.cfg)))
        return lvl, found[: sc.max_scenarios]


def complexity_key(c: Candidate, cfg: Settings) -> float:
    """Occam tie-breaker: fewer waves / simpler patterns first."""
    return float(cfg.guidelines.complexity.by_pattern.get(c.spec.type.value, 0.0)) + 0.01 * c.k


def dedupe(cands: list[Candidate], cfg: Settings) -> list[Candidate]:
    """Remove duplicates.

    1. Same pattern *family* on identical pivots (e.g. leading vs. ending
       diagonal, zigzag vs. double zigzag, flat variants): keep the best.
    2. Near-identical counts of the same type: all wave ends within
       ``dedup_bar_tol`` bars and ``dedup_price_tol`` relative price.
    """
    sc = cfg.search
    best: dict[tuple[str, tuple[int, ...], int, bool], Candidate] = {}
    for c in cands:
        key = (c.spec.family, c.positions, c.level, c.pullback)
        other = best.get(key)
        if other is None or (c.total, -complexity_key(c, cfg)) > (other.total, -complexity_key(other, cfg)):
            best[key] = c
    unique = sorted(best.values(), key=lambda c: c.total, reverse=True)
    kept: list[Candidate] = []
    buckets: dict[tuple[PatternType, int, bool], list[Candidate]] = {}
    for c in unique:
        bucket = buckets.setdefault((c.spec.type, c.k, c.pullback), [])
        if any(_near(c, o, sc.dedup_bar_tol, sc.dedup_price_tol) for o in bucket):
            continue
        bucket.append(c)
        kept.append(c)
    return kept


def _near(a: Candidate, b: Candidate, bar_tol: int, price_tol: float) -> bool:
    va, vb = a.view, b.view
    for ta, tb, pa, pb in zip(va.t, vb.t, va.raw, vb.raw):
        if abs(ta - tb) > bar_tol or abs(pa / pb - 1.0) > price_tol:
            return False
    return True
