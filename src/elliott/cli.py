"""Command line interface.

Examples::

    elliott analyze data.csv --threshold atr:2 --log auto --top 3 --out report/
    elliott analyze --ticker BTC-USD --interval 1d --period 2y
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from elliott import DISCLAIMER
from elliott.data.loader import DataValidationError, load_data, load_ticker, resample_ohlc
from elliott.fmt import fmt_price
from elliott.pipeline import analyze as run_analysis
from elliott.report import PROBABILITY_NOTE, write_reports
from elliott.settings import ThresholdSpec, load_settings, with_overrides

app = typer.Typer(add_completion=False, help="Elliott-Wellen-Analyzer – automatische Wellenzählung mit Szenarien.")


@app.callback()
def main() -> None:
    """Elliott-Wellen-Analyzer."""


@app.command()
def analyze(
    csv: Optional[Path] = typer.Argument(None, help="CSV-Datei (TradingView, Yahoo, deutsches Format …)"),
    ticker: Optional[str] = typer.Option(None, "--ticker", help="Ticker für yfinance, z. B. BTC-USD"),
    period: str = typer.Option("2y", help="yfinance-Zeitraum (z. B. 6mo, 2y, max)"),
    interval: str = typer.Option("1d", help="yfinance-Intervall (1h, 1d, 1wk …)"),
    resample: Optional[str] = typer.Option(None, help="Auf höhere Zeiteinheit aggregieren (z. B. 1wk)"),
    threshold: Optional[str] = typer.Option(None, help="ZigZag-Schwelle, z. B. atr:2 oder pct:5"),
    log: str = typer.Option("auto", "--log", help="auto | linear | log"),
    top: Optional[int] = typer.Option(None, help="Anzahl Szenarien (Top-N)"),
    start: Optional[str] = typer.Option(None, help="Startdatum der Zählung (YYYY-MM-DD)"),
    config: Optional[Path] = typer.Option(None, help="Eigene YAML-Konfiguration"),
    out: Optional[Path] = typer.Option(None, help="Ausgabeverzeichnis für JSON/Markdown/HTML"),
    cdn: bool = typer.Option(False, help="Plotly im HTML per CDN statt eingebettet laden"),
) -> None:
    """Zählung durchführen und Szenarien ausgeben."""
    if log not in ("auto", "linear", "log"):
        raise typer.BadParameter("--log muss auto, linear oder log sein")
    try:
        cfg = load_settings(config)
        if threshold:
            cfg = with_overrides(cfg, {"pivots": {"threshold": ThresholdSpec.parse(threshold).model_dump()}})
        if top:
            cfg = with_overrides(cfg, {"scoring": {"top_n": top}})
        if ticker:
            data = load_ticker(ticker, cfg.data, period=period, interval=interval)
        elif csv is not None:
            data = load_data(csv, cfg.data)
        else:
            raise typer.BadParameter("CSV-Datei oder --ticker angeben")
        if resample:
            data = resample_ohlc(data, resample, cfg.data)
        result = run_analysis(data, cfg, start=start, log_mode=log)
    except (DataValidationError, ValueError) as exc:
        typer.secho(f"Fehler: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    n = top or cfg.scoring.top_n
    typer.secho(f"⚠ {DISCLAIMER}", fg=typer.colors.YELLOW)
    for w in result.warnings:
        typer.echo(f"Hinweis: {w}")
    m = result.meta
    typer.echo(
        f"{m.source}: {m.bars} Kerzen, {'log' if m.log_scale else 'linear'}, letzter Kurs {fmt_price(m.last_close)}, "
        f"{len(result.scenarios)} Szenarien in {m.runtime_s:.2f} s"
    )
    typer.echo(PROBABILITY_NOTE)
    for s in result.scenarios[:n]:
        p = s.projection
        prob = f"{s.probability * 100:.0f} %" if s.probability is not None else "–"
        typer.secho(f"\n#{s.rank} {s.count.pattern.german} – Score {s.score.total:.1f} – Gewichtung {prob}", bold=True)
        if p is None:
            continue
        typer.echo(f"  Position:      {p.position}")
        if p.sub_position:
            typer.echo(f"  Subzählung:    {p.sub_position}")
        typer.echo(f"  Erwartet:      {p.next_move}")
        for z in p.targets:
            flag = " (Konfluenz)" if z.highlighted else ""
            typer.echo(f"  Ziel:          {fmt_price(z.low)} – {fmt_price(z.high)}{flag}: {', '.join(z.sources)}")
        typer.echo(f"  Invalidierung: {p.invalidation.reason}")
        typer.echo(f"  Alternative:   {p.alternative}")
    if out is not None:
        paths = write_reports(result, out, df=data.df, plot_cfg=cfg.plotting, top=n, embed_plotlyjs="cdn" if cdn else True)
        typer.echo("\nReports: " + ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":  # pragma: no cover
    app()
