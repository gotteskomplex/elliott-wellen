"""Reports: JSON (machine readable), Markdown and HTML (with embedded chart)."""

from __future__ import annotations

import html
import json
from pathlib import Path

import pandas as pd

from elliott.fmt import fmt_pct, fmt_price, fmt_ratio
from elliott.models import AnalysisResult, Scenario, WaveCount

PROBABILITY_NOTE = (
    "Die relative Gewichtung ist eine heuristische Softmax-Gewichtung der Scores der Top-Szenarien – "
    "keine statistische Wahrscheinlichkeit."
)


def to_json(result: AnalysisResult) -> str:
    """Complete result as JSON."""
    return result.model_dump_json(indent=2)


def _zones(s: Scenario) -> str:
    if s.projection is None or not s.projection.targets:
        return "–"
    parts = []
    for z in s.projection.targets:
        txt = f"{fmt_price(z.low)}–{fmt_price(z.high)}"
        parts.append(f"**{txt}**" if z.highlighted else txt)
    return "; ".join(parts)


def _prob(s: Scenario) -> str:
    return fmt_pct(s.probability) if s.probability is not None else "–"


RATIO_NAMES = {"vs_w1": "×W1", "vs_prev": "×Vorwelle", "vs_prev_same_dir": "×gleichgerichtete Vorwelle"}


def _wave_table(count: WaveCount) -> list[str]:
    lines = [
        "| Welle | Start | Ende | Länge | Dauer (Kerzen) | Verhältnisse | Binnenstruktur |",
        "|---|---|---|---|---|---|---|",
    ]
    for w in count.waves:
        ratios = ", ".join(f"{fmt_ratio(v)} {RATIO_NAMES.get(k, k)}" for k, v in w.ratios.items())
        sub = w.subwave_status.german + (f" – {w.subwave_note}" if w.subwave_note else "")
        lines.append(
            f"| {w.display_label}{' (läuft)' if w.open else ''} | {w.start.ts:%Y-%m-%d} @ {fmt_price(w.start.price)} "
            f"| {w.end.ts:%Y-%m-%d} @ {fmt_price(w.end.price)} | {fmt_price(w.length_price)} "
            f"| {w.duration_bars} | {ratios} | {sub} |"
        )
    return lines


def scenario_markdown(s: Scenario) -> str:
    """Detailed Markdown section for one scenario."""
    p = s.projection
    out = [f"### {s.id} – {s.count.pattern.german} ({s.count.direction.german}) – Score {s.score.total:.1f}", ""]
    if p is not None:
        out += [
            f"- **Aktuelle Position:** {p.position}",
            *( [f"- **Subzählung:** {p.sub_position}"] if p.sub_position else [] ),
            f"- **Erwartete Bewegung:** {p.next_move}",
            f"- **Kursziele ({p.target_wave}):** " + (
                "; ".join(
                    f"{fmt_price(z.low)}–{fmt_price(z.high)} ({', '.join(z.sources)}){' – **Konfluenz**' if z.highlighted else ''}"
                    for z in p.targets
                ) or "–"
            ),
            f"- **Invalidierung:** {p.invalidation.reason} (Regel: `{p.invalidation.rule}`)",
        ]
        others = [i for i in p.invalidations if i is not p.invalidation and i.reason != p.invalidation.reason]
        for inv in others:
            out.append(f"  - {'hart' if inv.hard else 'strukturell'}: {inv.reason}")
        out.append(f"- **Alternative:** {p.alternative}")
        if p.time_window is not None:
            out.append(
                f"- **Zeitfenster (unsicher):** {p.time_window.earliest:%Y-%m-%d} bis {p.time_window.latest:%Y-%m-%d} – {p.time_window.note}"
            )
    if s.notes:
        out.append("- **Hinweise:** " + " · ".join(s.notes))
    out += ["", "**Wellen**", "", *_wave_table(s.count), ""]
    out += ["**Score-Aufschlüsselung**", "", "| Richtlinie | Gewicht | Wert | Punkte | Begründung |", "|---|---|---|---|---|"]
    for g in s.score.guidelines:
        val = "n/a" if g.value is None else f"{g.value:.2f}"
        out.append(f"| {g.title} | {g.weight:g} | {val} | {g.points:.1f} | {g.reason} |")
    for pen in s.score.penalties:
        out.append(f"| Abzug: {pen.name} | – | – | −{pen.points:.1f} | {pen.reason} |")
    out.append(f"| **Gesamt** | | | **{s.score.total:.1f}** | Richtlinien {s.score.guideline_score:.1f} abzüglich Strafpunkte |")
    out += ["", "**Harte Regeln**", ""]
    for r in s.rule_checks:
        mark = "✅" if r.passed and not r.deferred else ("⏳" if r.deferred else "❌")
        out.append(f"- {mark} `{r.name}`: {r.reason}")
    out.append("")
    return "\n".join(out)


def to_markdown(result: AnalysisResult, top: int | None = None) -> str:
    """Complete Markdown report."""
    m = result.meta
    scenarios = result.scenarios[:top] if top else result.scenarios
    out = [
        "# Elliott-Wellen-Analyse",
        "",
        f"> ⚠️ **{result.disclaimer}**",
        "",
        f"- Quelle: {m.source}",
        f"- Zeitraum: {m.start:%Y-%m-%d} bis {m.end:%Y-%m-%d} ({m.bars} Kerzen), letzter Schlusskurs {fmt_price(m.last_close)}",
        f"- Skala: {'logarithmisch' if m.log_scale else 'linear'} · Suchgrad: {m.levels[m.search_level].name if m.levels else '–'}",
        "- Grade: " + ", ".join(f"{lv.name} ({lv.threshold}, {lv.pivot_count} Pivots)" for lv in m.levels),
        f"- Laufzeit: {m.runtime_s:.2f} s, {m.states_explored} Suchzustände"
        + (" – **Suchbudget erschöpft**" if m.budget_exhausted else ""),
        "",
    ]
    if result.warnings:
        out += ["**Hinweise zu den Daten:**", "", *[f"- {w}" for w in result.warnings], ""]
    out += [
        "## Szenarien",
        "",
        f"_{PROBABILITY_NOTE}_",
        "",
        "| Rang | Muster | Aktuelle Position | Score | rel. Gewichtung | Ziele | Invalidierung |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in scenarios:
        p = s.projection
        inv = f"{'<' if p.invalidation.side == 'below' else '>'} {fmt_price(p.invalidation.price)}" if p else "–"
        out.append(
            f"| {s.rank} | {s.count.pattern.german} | {p.position if p else '–'} | {s.score.total:.1f} | {_prob(s)} "
            f"| {_zones(s)} | {inv} |"
        )
    out += ["", "## Details", ""]
    for s in scenarios:
        out.append(scenario_markdown(s))
    out += ["---", f"_{result.disclaimer}_", ""]
    return "\n".join(out)


def _md_to_html(md: str) -> str:
    """Minimal Markdown -> HTML for the report (tables, lists, headings, emphasis)."""
    import re

    def inline(text: str) -> str:
        t = html.escape(text)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
        t = re.sub(r"`(.+?)`", r"<code>\1</code>", t)
        t = re.sub(r"(?<![\w*])_(.+?)_(?![\w*])", r"<em>\1</em>", t)
        return t

    lines, out, i = md.splitlines(), [], 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows if not set(r) <= set("|-: ")]
            out.append("<table>")
            for j, row in enumerate(cells):
                tag = "th" if j == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{inline(c)}</{tag}>" for c in row) + "</tr>")
            out.append("</table>")
            continue
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            out.append(f"<h{level}>{inline(line[level:].strip())}</h{level}>")
        elif line.startswith("> "):
            out.append(f"<div class='disclaimer'>{inline(line[2:])}</div>")
        elif line.lstrip().startswith("- "):
            items = []
            while i < len(lines) and lines[i].lstrip().startswith("- "):
                nested = lines[i].startswith("  ")
                items.append(f"<li{' class=nested' if nested else ''}>{inline(lines[i].lstrip()[2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip() == "---":
            out.append("<hr>")
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)


HTML_STYLE = """
:root { --surface: #fcfcfb; --text: #0b0b0b; --muted: #52514e; --border: #e4e3de; --warn-bg: #fff4e5; }
@media (prefers-color-scheme: dark) { :root { --surface: #1a1a19; --text: #ffffff; --muted: #c3c2b7; --border: #3a3a37; --warn-bg: #3a2a10; } }
body { font-family: system-ui, -apple-system, Segoe UI, sans-serif; background: var(--surface); color: var(--text);
       max-width: 1200px; margin: 0 auto; padding: 16px; line-height: 1.45; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 16px; font-size: 0.9em; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--border); padding: 4px 8px; text-align: left; vertical-align: top; }
th { background: rgba(127,127,127,0.08); }
.disclaimer { background: var(--warn-bg); border-left: 4px solid #eda100; padding: 8px 12px; margin: 12px 0; }
li.nested { margin-left: 1.5em; color: var(--muted); }
code { font-size: 0.85em; }
"""


def to_html(result: AnalysisResult, figure_html: str | None = None, top: int | None = None) -> str:
    """Standalone HTML report; ``figure_html`` is a Plotly ``to_html`` fragment."""
    body = _md_to_html(to_markdown(result, top=top))
    chart = f"<div class='chart'>{figure_html}</div>" if figure_html else ""
    first_h2 = body.find("<h2>")
    body = body[:first_h2] + chart + body[first_h2:] if first_h2 >= 0 else chart + body
    return (
        "<!doctype html><html lang='de'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        f"<title>Elliott-Wellen-Analyse</title><style>{HTML_STYLE}</style></head><body>{body}</body></html>"
    )


def write_reports(
    result: AnalysisResult,
    out_dir: str | Path,
    df: pd.DataFrame | None = None,
    plot_cfg=None,  # type: ignore[no-untyped-def]
    top: int | None = None,
    embed_plotlyjs: bool | str = True,
) -> dict[str, Path]:
    """Write JSON, Markdown and HTML reports; returns the file paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "analysis.json", "markdown": out / "report.md", "html": out / "report.html"}
    paths["json"].write_text(to_json(result), encoding="utf-8")
    paths["markdown"].write_text(to_markdown(result, top=top), encoding="utf-8")
    fig_html = None
    if df is not None and plot_cfg is not None:
        from elliott.plotting import plot_analysis  # presentation layer, imported lazily

        ids = [s.id for s in (result.scenarios[:top] if top else result.scenarios)]
        fig = plot_analysis(df, result, plot_cfg, scenario_ids=ids)
        fig_html = fig.to_html(full_html=False, include_plotlyjs=embed_plotlyjs)
    paths["html"].write_text(to_html(result, fig_html, top=top), encoding="utf-8")
    return paths


def load_json(path: str | Path) -> AnalysisResult:
    """Read a JSON report back into an :class:`AnalysisResult`."""
    return AnalysisResult.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))
