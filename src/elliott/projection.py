"""Projection: current position, price targets, invalidation levels, alternatives.

Invalidation levels are derived *from the hard rules themselves*: the count is
extended hypothetically (the current wave extends further, or the next wave
starts) and a bisection finds the exact price at which a rule of
:mod:`elliott.rules` starts to fail. The failing rule is reported with it.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from elliott.fmt import fmt_price, fmt_ratio
from elliott.labels import display_label
from elliott.measure import CountView, PriceScale
from elliott.models import Direction, Invalidation, PatternType, Projection, TargetZone, TimeWindow
from elliott.patterns import PatternSpec
from elliott.rules import first_failure
from elliott.search import Candidate, SearchEngine
from elliott.settings import Settings

# Typical form of the next wave (descriptive text, not thresholds).
_EXPECTED_FORM: dict[tuple[str, int], str] = {
    ("impulse", 1): "Impuls (5 Wellen) oder Leading Diagonal",
    ("impulse", 2): "scharfe Korrektur (oft Zigzag), typisch 50–61,8 % von W1",
    ("impulse", 3): "Impuls (5 Wellen), meist die stärkste Welle",
    ("impulse", 4): "Seitwärtskorrektur (oft Flat oder Dreieck), typisch 23,6–38,2 % von W3",
    ("impulse", 5): "Impuls (5 Wellen) oder Ending Diagonal",
    ("diagonal", 1): "3- oder 5-teilige Welle",
    ("diagonal", 2): "Zigzag",
    ("diagonal", 3): "3- oder 5-teilige Welle",
    ("diagonal", 4): "Zigzag mit Überlappung zu Welle 1",
    ("diagonal", 5): "3- oder 5-teilige Welle",
    ("zigzag", 2): "3-teilige Korrektur (B)",
    ("zigzag", 3): "Impuls (5 Wellen) oder Ending Diagonal (C)",
    ("zigzag", 4): "3-teilige Verbindungswelle (X)",
    ("zigzag", 5): "Zigzag (Z)",
    ("flat", 2): "3-teilige Welle B, mindestens ca. 90 % von A",
    ("flat", 3): "Impuls (5 Wellen) oder Ending Diagonal (C)",
    ("triangle", 2): "3-teilige Welle (meist Zigzag)",
    ("triangle", 3): "3-teilige Welle, kürzer als A",
    ("triangle", 4): "3-teilige Welle, kürzer als B",
    ("triangle", 5): "3-teilige Welle, kürzer als C – danach Thrust",
    ("combination", 2): "3-teilige Verbindungswelle (X)",
    ("combination", 3): "Zigzag, Flat oder Dreieck (Y)",
    ("combination", 4): "3-teilige Verbindungswelle (X)",
    ("combination", 5): "Zigzag, Flat oder Dreieck (Z)",
}

_AFTER_FORM: dict[str, str] = {
    "impulse": "Korrektur des gesamten Impulses (3-teilig)",
    "leading_diagonal": "tiefe Korrektur (Welle 2/B), danach Fortsetzung",
    "ending_diagonal": "scharfe Umkehr, meist bis zum Start des Diagonals",
    "triangle": "Thrust in Richtung des übergeordneten Trends",
    "correction": "impulsive Bewegung in Richtung des übergeordneten Trends",
}


@dataclass
class _Target:
    price: float
    source: str
    priority: int


# ---------------------------------------------------------------------------
# targets
# ---------------------------------------------------------------------------


def _ext(v: CountView, origin_x: float, length: float, sign: int, ratios: list[float], text: str) -> list[tuple[float, str, int]]:
    """Normalized target values ``origin + sign * r * length``."""
    return [(origin_x + sign * r * length, text.format(r=fmt_ratio(r)), i) for i, r in enumerate(ratios)]


def wave_targets(spec: PatternSpec, v: CountView, wave: int, prior: float | None, cfg: Settings) -> list[tuple[float, str, int]]:
    """Normalized target values for the end of ``wave`` (1-based, may be v.k+1).

    Returns ``(x, source text, priority)``; lower priority = more typical ratio.
    """
    pc = cfg.projection
    fam = spec.family
    up = 1 if wave % 2 == 1 else -1  # normalized direction of the wave
    start = v.x[wave - 1]
    L = v.L

    if wave == 1:
        if prior:
            return _ext(v, start, prior, up, pc.t("first_wave"), "{r} × vorherige Gegenbewegung")
        return []
    if fam in ("impulse", "diagonal"):
        if fam == "impulse":
            if wave == 2:
                return _ext(v, v.x[1], L(1), -1, pc.t("impulse_w2"), "{r} Retracement von W1")
            if wave == 3:
                return _ext(v, start, L(1), 1, pc.t("impulse_w3"), "{r} × W1")
            if wave == 4:
                return _ext(v, v.x[3], L(3), -1, pc.t("impulse_w4"), "{r} Retracement von W3")
            out = _ext(v, start, L(1), 1, pc.t("impulse_w5_w1"), "{r} × W1")
            base = v.x[3] - v.x[0]
            out += _ext(v, start, base, 1, pc.t("impulse_w5_w1w3"), "{r} × (W1 bis W3)")
            return out
        if wave in (2, 4):
            return _ext(v, v.x[wave - 1], L(wave - 1), -1, pc.t("diagonal_retrace"), "{r} Retracement von W%d" % (wave - 1))
        expanding = v.has(3) and v.k >= 3 and L(3) > L(1)
        key = "diagonal_next_expanding" if expanding else "diagonal_next_contracting"
        return _ext(v, start, L(wave - 2), 1, pc.t(key), "{r} × W%d" % (wave - 2))
    if fam == "zigzag":
        if wave % 2 == 0:
            key = "zigzag_b" if spec.type is PatternType.ZIGZAG else "combination_x"
            return _ext(v, v.x[wave - 1], L(wave - 1), -1, pc.t(key), "{r} Retracement von %s" % spec.labels[wave - 2])
        key = "zigzag_c" if spec.type is PatternType.ZIGZAG else "combination_next"
        return _ext(v, start, L(wave - 2), 1, pc.t(key), "{r} × %s" % spec.labels[wave - 3])
    if fam == "flat":
        if wave == 2:
            return _ext(v, v.x[1], L(1), -1, pc.t("flat_b"), "B = {r} × A")
        return _ext(v, start, L(1), 1, pc.t("flat_c"), "C = {r} × A")
    if fam == "triangle":
        if wave == 2:
            return _ext(v, v.x[1], L(1), -1, pc.t("triangle_first"), "B = {r} × A")
        return _ext(v, start, L(wave - 2), up, pc.t("triangle_next"), "{r} × %s" % spec.labels[wave - 3])
    # combination
    if wave % 2 == 0:
        return _ext(v, v.x[wave - 1], L(wave - 1), -1, pc.t("combination_x"), "{r} Retracement von %s" % spec.labels[wave - 2])
    return _ext(v, start, L(wave - 2), 1, pc.t("combination_next"), "{r} × %s" % spec.labels[wave - 3])


def after_targets(spec: PatternSpec, v: CountView, cfg: Settings) -> tuple[list[tuple[float, str, int]], int, str]:
    """Targets for the move after a complete pattern -> (targets, normalized direction, form)."""
    pc = cfg.projection
    net = abs(v.x[-1] - v.x[0])
    end = v.x[-1]
    if spec.type is PatternType.IMPULSE:
        return _ext(v, end, net, -1, pc.t("after_motive"), "{r} Retracement des Impulses"), -1, _AFTER_FORM["impulse"]
    if spec.type is PatternType.LEADING_DIAGONAL:
        return _ext(v, end, net, -1, pc.t("after_motive"), "{r} Retracement des Diagonals"), -1, _AFTER_FORM["leading_diagonal"]
    if spec.type is PatternType.ENDING_DIAGONAL:
        return _ext(v, end, net, -1, pc.t("after_ending_diagonal"), "{r} Retracement des Diagonals"), -1, _AFTER_FORM["ending_diagonal"]
    if spec.family == "triangle":
        return _ext(v, end, v.L(1), -1, pc.t("after_triangle"), "Thrust {r} × A (breiteste Stelle)"), -1, _AFTER_FORM["triangle"]
    return _ext(v, end, net, -1, pc.t("after_correction"), "{r} × Länge der Korrektur"), -1, _AFTER_FORM["correction"]


def cluster_targets(targets: list[_Target], current: float, direction: int, cfg: Settings) -> list[TargetZone]:
    """Merge nearby targets into zones; zones with several sources = confluence."""
    pc = cfg.projection
    ahead = [t for t in targets if (t.price - current) * direction > 0]
    pool = ahead or sorted(targets, key=lambda t: -abs(t.price - current))[:1]
    pool.sort(key=lambda t: t.price)
    groups: list[list[_Target]] = []
    for t in pool:
        if groups and abs(t.price / groups[-1][-1].price - 1.0) <= pc.confluence_tol_rel:
            groups[-1].append(t)
        else:
            groups.append([t])
    zones: list[tuple[tuple[int, int, float], TargetZone]] = []
    for g in groups:
        lo, hi = min(t.price for t in g), max(t.price for t in g)
        center = sum(t.price for t in g) / len(g)
        hw = pc.zone_half_width_rel * center
        sources = list(dict.fromkeys(t.source for t in g))
        zone = TargetZone(
            low=lo - hw, high=hi + hw, center=center, sources=sources,
            confluence=len(sources), highlighted=len(sources) >= 2,
        )
        key = (-len(sources), min(t.priority for t in g), abs(center - current))
        zones.append((key, zone))
    zones.sort(key=lambda kz: kz[0])
    return [z for _, z in zones[: pc.max_zones]]


# ---------------------------------------------------------------------------
# invalidation (bisection over hard rules)
# ---------------------------------------------------------------------------


def _bisect_limit(
    spec: PatternSpec,
    build: Callable[[float], CountView],
    ref: float,
    sign: int,
    span: float,
    bound: float | None,
    cfg: Settings,
) -> tuple[float, str, str] | None:
    """Find the first normalized value beyond ``ref`` (direction ``sign``) at
    which a hard rule fails. Returns ``(x, rule_key, reason)`` or ``None``."""
    rules = spec.all_rules

    def fails(x: float) -> tuple[bool, str, str]:
        res = first_failure(rules, build(x), cfg.rules)
        if res is None:
            return False, "", ""
        return True, res[0].key, res[1].reason

    far = ref + sign * span
    if bound is not None:
        far = min(far, bound) if sign > 0 else max(far, bound)
    tiny = max(1e-12, abs(span) * 1e-9)
    f_near = fails(ref + sign * tiny)
    if f_near[0]:
        return ref, f_near[1], f_near[2]
    if not fails(far)[0]:
        return None
    lo, hi = ref, far
    for _ in range(cfg.projection.invalidation_iterations):
        mid = (lo + hi) / 2.0
        if fails(mid)[0]:
            hi = mid
        else:
            lo = mid
    _, key, reason = fails(hi)
    return hi, key, reason


def _to_price(x: float, d: int, scale: PriceScale) -> float:
    return scale.inv(d * x)


def invalidations(
    spec: PatternSpec, v: CountView, running: bool, now_bar: int, scale: PriceScale, cfg: Settings
) -> list[Invalidation]:
    """Hard invalidation levels (rule based) plus structural levels."""
    d = v.d
    span = cfg.projection.invalidation_span * max(1e-9, max(v.x) - min(v.x))
    # linear prices must stay positive: normalized bound
    bound_up = None if scale.log or d > 0 else -1e-9  # d=-1: x = -price < 0
    bound_down = None if scale.log or d < 0 else 1e-9  # d=+1: x = price > 0
    k = v.k
    out: list[Invalidation] = []

    def add(res: tuple[float, str, str] | None, sign: int, what: str) -> None:
        if res is None:
            return
        x, key, reason = res
        price = _to_price(x, d, scale)
        side = "above" if sign * d > 0 else "below"
        word = "Über" if side == "above" else "Unter"
        out.append(Invalidation(price=price, side=side, rule=key, reason=f"{word} {fmt_price(price)} ist {what} ungültig: {reason}"))

    def extend_last(x: float) -> CountView:
        xs = v.x[:-1] + (x,)
        ts = v.t[:-1] + (now_bar,)
        return v.with_points(xs, ts, tuple(_to_price(val, d, scale) for val in xs), last_open=True)

    wave_sign = 1 if v.wave_up(k) else -1
    add(_bisect_limit(spec, extend_last, v.x[k], wave_sign, span, bound_up if wave_sign > 0 else bound_down, cfg),
        wave_sign, f"diese Zählung (Welle {spec.labels[k - 1]} läuft zu weit)")

    if running:
        if k >= 2:
            def cancel_last(x: float) -> CountView:
                xs = v.x[: k - 1] + (x,)
                ts = v.t[: k - 1] + (now_bar,)
                return v.with_points(xs, ts, tuple(_to_price(val, d, scale) for val in xs), last_open=True)

            prev_sign = -wave_sign
            add(_bisect_limit(spec, cancel_last, v.x[k - 1], prev_sign, span, bound_up if prev_sign > 0 else bound_down, cfg),
                prev_sign, "die Gesamtzählung")
        elif running:
            price = _to_price(v.x[0], d, scale)
            side = "below" if d > 0 else "above"
            out.append(Invalidation(price=price, side=side, rule="start_extreme",
                                    reason=f"{'Unter' if side == 'below' else 'Über'} {fmt_price(price)} wird der Start von Welle {spec.labels[0]} verletzt"))
    elif k < spec.n:
        def add_next(x: float) -> CountView:
            xs = v.x + (x,)
            ts = v.t + (now_bar,)
            return v.with_points(xs, ts, tuple(_to_price(val, d, scale) for val in xs), last_open=True)

        next_sign = -wave_sign
        add(_bisect_limit(spec, add_next, v.x[k], next_sign, span, bound_up if next_sign > 0 else bound_down, cfg),
            next_sign, f"die Zählung (Welle {spec.labels[k]} läuft zu weit)")

    # structural (soft) levels
    if running and k >= 1:
        price = _to_price(v.x[k - 1], d, scale)
        side = "below" if wave_sign * d > 0 else "above"
        out.append(Invalidation(
            price=price, side=side, rule="wave_start", hard=False,
            reason=f"{'Unter' if side == 'below' else 'Über'} {fmt_price(price)} hätte Welle {spec.labels[k - 1]} noch nicht begonnen (Relabeling nötig)",
        ))
    else:
        price = _to_price(v.x[k], d, scale)
        side = "above" if wave_sign * d > 0 else "below"
        nxt = f"Welle {spec.labels[k]}" if k < spec.n else "die Folgebewegung"
        out.append(Invalidation(
            price=price, side=side, rule="wave_extreme", hard=False,
            reason=f"{'Über' if side == 'above' else 'Unter'} {fmt_price(price)} läuft Welle {spec.labels[k - 1]} weiter – {nxt} hat dann noch nicht begonnen",
        ))
    return out


def primary_invalidation(levels: list[Invalidation], current: float) -> Invalidation:
    """Nearest hard level; structural level only if no hard level exists."""
    hard = [lv for lv in levels if lv.hard]
    pool = hard or levels
    return min(pool, key=lambda lv: abs(lv.price - current))


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------


def _bar_time(index: pd.DatetimeIndex, bars_ahead: float) -> datetime:
    step = index.to_series().diff().median() if len(index) > 1 else pd.Timedelta(days=1)
    return (index[-1] + step * max(0.0, bars_ahead)).round("s").to_pydatetime()


def project(
    cand: Candidate,
    engine: SearchEngine,
    df: pd.DataFrame,
    scale: PriceScale,
    cfg: Settings,
    degree: int,
) -> tuple[Projection, Candidate | None]:
    """Build the projection of a candidate. Returns it plus the current subcount."""
    spec, v = cand.spec, cand.view
    k, n, d = v.k, spec.n, v.d
    arr = engine.arrays[cand.level]
    fine = engine.arrays[-1]
    current = float(df["close"].iloc[-1])
    now_bar = len(df) - 1
    last_bar = int(arr.bars[cand.positions[-1]])
    lab = lambda w: display_label(spec, w, degree, cfg.plotting)  # noqa: E731

    # --- which wave is running? -------------------------------------------
    fine_after = int((fine.bars > last_bar).sum())
    running = v.running
    sub = None
    if not running and fine_after == 0:
        sub = engine.current_subcount(cand)
        if sub is not None and (sub.k < sub.spec.n or sub.view.running):
            running = True
    if running and sub is None:
        sub = engine.current_subcount(cand)

    # --- targets -------------------------------------------------------------
    if running:
        target_wave = k
        raw_targets = wave_targets(spec, v, k, v.prior, cfg)
        move_sign = 1 if v.wave_up(k) else -1
        form = _EXPECTED_FORM.get((spec.family, k), "")
        target_label = lab(k)
    elif k < n:
        target_wave = k + 1
        raw_targets = wave_targets(spec, v, k + 1, v.prior, cfg)
        move_sign = 1 if v.wave_up(k + 1) else -1
        form = _EXPECTED_FORM.get((spec.family, k + 1), "")
        target_label = lab(k + 1)
    else:
        target_wave = n + 1
        raw_targets, move_sign, form = after_targets(spec, v, cfg)
        target_label = "Folgebewegung"

    # Drop targets that would break a hard rule (only inside the pattern).
    targets: list[_Target] = []
    for x, text, prio in raw_targets:
        if target_wave <= n:
            xs = v.x[: target_wave] + (x,)
            ts = v.t[: target_wave] + (now_bar,)
            test = v.with_points(xs, ts, tuple(_to_price(val, d, scale) for val in xs), last_open=True)
            if first_failure(spec.all_rules, test, cfg.rules) is not None:
                continue
        if not scale.log and d * x <= 0:
            continue
        targets.append(_Target(_to_price(x, d, scale), text, prio))
    price_dir = move_sign * d
    zones = cluster_targets(targets, current, price_dir, cfg) if targets else []

    # --- invalidation -------------------------------------------------------
    levels = invalidations(spec, v, running, now_bar, scale, cfg)
    primary = primary_invalidation(levels, current) if levels else Invalidation(
        price=current, side="below", rule="none", reason="Keine Invalidierung bestimmbar", hard=False
    )

    # --- texts --------------------------------------------------------------
    direction = Direction.from_sign(price_dir)
    last_price = float(arr.price[cand.positions[-1]])
    if running:
        position = f"{spec.type.german}: Welle {lab(k)} ({Direction.from_sign((1 if v.wave_up(k) else -1) * d).german}) läuft noch"
        next_move = f"Fortsetzung von Welle {lab(k)} {direction.german} – {form}".rstrip(" –")
    elif k < n:
        position = (
            f"{spec.type.german}: Welle {lab(k)} evtl. abgeschlossen bei {fmt_price(last_price)} "
            f"(provisorisch) – als Nächstes Welle {lab(k + 1)}"
        )
        next_move = f"Welle {lab(k + 1)} {direction.german} – {form}"
    else:
        position = f"{spec.type.german} evtl. vollständig (Welle {lab(n)} bei {fmt_price(last_price)}, provisorisch)"
        next_move = f"{form} ({direction.german})"

    sub_position = ""
    if sub is not None:
        sub_lab = display_label(sub.spec, sub.k, degree + 1, cfg.plotting)
        state = "läuft" if (sub.view.running or sub.k < sub.spec.n) else "evtl. abgeschlossen"
        owner = lab(k) if running else (lab(k + 1) if k < n else "Folgebewegung")
        sub_position = (
            f"Subzählung (feinerer Grad) von Welle {owner if running else lab(k)}: {sub.spec.type.german}, "
            f"{sub.k} von {sub.spec.n} Wellen – Subwelle {sub_lab} {state}"
        )

    # --- time window --------------------------------------------------------
    time_window = None
    ref_wave = None
    if target_wave <= n and target_wave >= 3:
        ref_wave = target_wave - 2
    elif k >= 1:
        ref_wave = k
    if ref_wave is not None and ref_wave <= k:
        ref_dur = v.dur(ref_wave)
        ratios = cfg.projection.time_ratios
        start_bar = int(v.t[k - 1]) if running else int(v.t[k])
        elapsed = now_bar - start_bar
        lo_b = min(ratios) * ref_dur - elapsed
        hi_b = max(ratios) * ref_dur - elapsed
        if hi_b > 0:
            time_window = TimeWindow(
                earliest=_bar_time(df.index, lo_b), latest=_bar_time(df.index, hi_b),
                note=(
                    f"Fib-Zeitprojektion ({'/'.join(fmt_ratio(r) for r in ratios)} × Dauer Welle {lab(ref_wave)}) – "
                    "sehr unsicher, nur als grobe Orientierung"
                ),
            )

    path: list[tuple[datetime, float]] = []
    if zones:
        idx = min(cfg.projection.path_ratio_index, len(zones) - 1)
        goal = zones[idx].center
        bars_ahead = (v.dur(ref_wave) if ref_wave is not None and ref_wave <= k else max(5, v.dur(k)))
        start_bar = int(v.t[k - 1]) if running else int(v.t[k])
        remaining = max(1.0, bars_ahead - (now_bar - start_bar))
        path = [(df.index[-1].to_pydatetime(), current), (_bar_time(df.index, remaining), goal)]

    proj = Projection(
        position=position,
        current_wave=lab(k) if running else (lab(k + 1) if k < n else "–"),
        current_wave_running=running,
        next_move=next_move,
        next_direction=direction,
        expected_pattern=form,
        targets=zones,
        invalidation=primary,
        invalidations=levels,
        time_window=time_window,
        path=path,
        sub_position=sub_position,
        target_wave=target_label,
    )
    return proj, sub


def is_invalidated_at(levels: list[Invalidation], price: float) -> bool:
    """True if ``price`` breaks one of the hard levels."""
    for lv in levels:
        if not lv.hard:
            continue
        if lv.side == "below" and price < lv.price:
            return True
        if lv.side == "above" and price > lv.price:
            return True
    return False


def nudge_beyond(inv: Invalidation) -> float:
    """A price just beyond an invalidation level."""
    eps = max(abs(inv.price) * 1e-6, 1e-12)
    return inv.price - eps if inv.side == "below" else inv.price + eps


def describe_ratio(value: float) -> str:  # pragma: no cover - convenience
    return fmt_ratio(value) if math.isfinite(value) else "–"
