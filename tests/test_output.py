"""Plotly chart, reports and CLI."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from elliott.cli import app
from elliott.data.loader import frame_to_market_data
from elliott.pipeline import analyze
from elliott.plotting import plot_analysis
from elliott.report import load_json, to_html, to_json, to_markdown, write_reports
from tests.synthetic import make_pattern


@pytest.fixture(scope="module")
def analysed(settings):
    syn = make_pattern("impulse", bars=300, noise=0.005, seed=2)
    data = frame_to_market_data(syn.df, settings.data)
    return data, analyze(data, settings)


def test_plot(analysed, settings) -> None:
    data, res = analysed
    fig = plot_analysis(data.df, res, settings.plotting, scenario_ids=[s.id for s in res.scenarios[:3]])
    names = [t.name for t in fig.data if t.name]
    assert any(n.startswith("S1") for n in names)
    assert fig.layout.annotations  # wave labels
    visible = [t.visible for t in fig.data if t.name and t.name.startswith("S2")]
    assert visible == ["legendonly"]  # alternatives toggleable
    log_fig = plot_analysis(data.df, res, settings.plotting, log_axis=True, show_volume=False)
    assert log_fig.layout.yaxis.type == "log"


def test_json_roundtrip(analysed, tmp_path) -> None:
    _, res = analysed
    text = to_json(res)
    assert json.loads(text)["scenarios"][0]["id"] == "S1"
    p = tmp_path / "a.json"
    p.write_text(text, encoding="utf-8")
    assert load_json(p).scenarios[0].count.pattern == res.scenarios[0].count.pattern


def test_markdown_and_html(analysed) -> None:
    _, res = analysed
    md = to_markdown(res, top=3)
    assert "Keine Anlageberatung" in md
    assert "| Rang | Muster" in md and "Score-Aufschlüsselung" in md and "Invalidierung" in md
    assert "keine statistische Wahrscheinlichkeit" in md
    html = to_html(res, "<div id='chart'></div>", top=3)
    assert html.startswith("<!doctype html>") and "<table>" in html and "id='chart'" in html
    assert "Keine Anlageberatung" in html


def test_write_reports(analysed, settings, tmp_path) -> None:
    data, res = analysed
    paths = write_reports(res, tmp_path, df=data.df, plot_cfg=settings.plotting, top=3, embed_plotlyjs="cdn")
    assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())
    assert "plotly" in paths["html"].read_text(encoding="utf-8").lower()


def test_cli_csv(tmp_path) -> None:
    syn = make_pattern("impulse", bars=300, noise=0.005, seed=2)
    csv = tmp_path / "data.csv"
    syn.df.to_csv(csv, index_label="Date")
    out = tmp_path / "report"
    result = CliRunner().invoke(
        app, ["analyze", str(csv), "--threshold", "atr:2", "--log", "auto", "--top", "3", "--out", str(out), "--cdn"]
    )
    assert result.exit_code == 0, result.output
    assert "Keine Anlageberatung" in result.output and "Invalidierung" in result.output
    assert (out / "report.html").exists() and (out / "analysis.json").exists()


def test_cli_errors(tmp_path) -> None:
    runner = CliRunner()
    assert runner.invoke(app, ["analyze"]).exit_code != 0
    bad = tmp_path / "bad.csv"
    bad.write_text("a,b\n1,2\n", encoding="utf-8")
    res = runner.invoke(app, ["analyze", str(bad)])
    assert res.exit_code == 1 and "Fehler" in res.output


def test_cli_ticker_uses_yfinance(monkeypatch, tmp_path) -> None:
    syn = make_pattern("zigzag", bars=200, prefix=2.0)
    frame = syn.df.rename(columns=str.title)

    class FakeTicker:
        def __init__(self, name: str) -> None:
            self.name = name

        def history(self, **kwargs):  # noqa: ANN003
            return frame

    import yfinance

    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    res = CliRunner().invoke(app, ["analyze", "--ticker", "BTC-USD", "--interval", "1d", "--period", "2y"])
    assert res.exit_code == 0, res.output
    assert "yfinance:BTC-USD" in res.output
