"""Streamlit web app for the Elliott wave analyzer.

Start with::

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from elliott import DISCLAIMER
from elliott.data.loader import DataValidationError, MarketData, load_data, load_ticker, resample_ohlc
from elliott.fmt import fmt_price
from elliott.models import AnalysisResult
from elliott.pipeline import analyze
from elliott.plotting import plot_analysis, plot_pivots
from elliott.pivots.zigzag import build_hierarchy
from elliott.report import PROBABILITY_NOTE, scenario_markdown, to_html, to_json, to_markdown
from elliott.settings import Settings, default_config_dict, deep_merge

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

st.set_page_config(page_title="Elliott-Wellen-Analyzer", page_icon="🌊", layout="wide")


# ---------------------------------------------------------------------------
# cached helpers
# ---------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def _load_csv(content: bytes, data_cfg: dict) -> MarketData:
    settings = Settings.model_validate(deep_merge(default_config_dict(), {"data": data_cfg}))
    return load_data(content, settings.data)


@st.cache_data(show_spinner=False, ttl=3600)
def _load_ticker(ticker: str, period: str, interval: str, data_cfg: dict) -> MarketData:
    settings = Settings.model_validate(deep_merge(default_config_dict(), {"data": data_cfg}))
    return load_ticker(ticker, settings.data, period=period, interval=interval)


@st.cache_data(show_spinner=False)
def _analyze(df: pd.DataFrame, source: str, cfg_dict: dict, start: date | None, log_mode: str, top: int) -> AnalysisResult:
    settings = Settings.model_validate(cfg_dict)
    data = MarketData(df=df, source=source)
    return analyze(data, settings, start=pd.Timestamp(start) if start else None, top_n=top, log_mode=log_mode)


# ---------------------------------------------------------------------------
# sidebar
# ---------------------------------------------------------------------------

defaults = default_config_dict()
overrides: dict = {}

st.sidebar.title("🌊 Elliott-Wellen")
source_kind = st.sidebar.radio("Datenquelle", ["Beispieldaten", "CSV-Upload", "Ticker (yfinance)"], index=0)

data: MarketData | None = None
error: str | None = None
try:
    if source_kind == "CSV-Upload":
        upload = st.sidebar.file_uploader(
            "CSV-Datei (TradingView, Yahoo, deutsches Format …)", type=["csv", "txt"],
            help="Spalten Date/Open/High/Low/Close (Volume optional). Screenshots werden nicht unterstützt.",
        )
        if upload is not None:
            data = _load_csv(upload.getvalue(), defaults["data"])
            data.source = upload.name
    elif source_kind == "Ticker (yfinance)":
        ticker = st.sidebar.text_input("Ticker", "BTC-USD")
        c1, c2 = st.sidebar.columns(2)
        period = c1.selectbox("Zeitraum", ["6mo", "1y", "2y", "5y", "10y", "max"], index=2)
        interval = c2.selectbox("Intervall", ["1h", "1d", "1wk", "1mo"], index=1)
        if ticker and st.sidebar.button("Daten laden", type="primary"):
            st.session_state["ticker_args"] = (ticker, period, interval)
        if "ticker_args" in st.session_state:
            data = _load_ticker(*st.session_state["ticker_args"], defaults["data"])
    else:
        files = sorted(EXAMPLES.glob("*.csv"))
        choice = st.sidebar.selectbox("Beispiel", [f.name for f in files]) if files else None
        if choice:
            data = _load_csv((EXAMPLES / choice).read_bytes(), defaults["data"])
            data.source = choice
except DataValidationError as exc:
    error = str(exc)

resample = st.sidebar.selectbox("Zeiteinheit", ["Original", "1wk", "1mo", "4h"], index=0,
                                help="Auf höhere Zeiteinheit aggregieren")

st.sidebar.subheader("Wendepunkte (ZigZag)")
mode = st.sidebar.radio("Schwelle", ["ATR-Vielfaches", "Prozent"], horizontal=True)
if mode == "ATR-Vielfaches":
    thr_value = st.sidebar.slider("× ATR(14)", 0.5, 8.0, float(defaults["pivots"]["threshold"]["value"]), 0.25)
    overrides.setdefault("pivots", {})["threshold"] = {"mode": "atr", "value": thr_value}
else:
    thr_value = st.sidebar.slider("Prozent", 0.5, 30.0, 5.0, 0.5)
    overrides.setdefault("pivots", {})["threshold"] = {"mode": "percent", "value": thr_value}
grade_preset = st.sidebar.selectbox(
    "Grade (Multiplikatoren grob/mittel/fein)", ["2 / 1 / 0,5", "3 / 1 / 0,5", "2 / 1 / 0,33", "4 / 2 / 1"], index=0
)
overrides["pivots"]["degree_multipliers"] = [float(x.replace(",", ".")) for x in grade_preset.split(" / ")]

st.sidebar.subheader("Analyse")
log_mode = st.sidebar.selectbox("Skala für Längen/Fibonacci", ["auto", "linear", "log"], index=0,
                                help="auto: logarithmisch, wenn Hoch/Tief > 3")
top_n = st.sidebar.slider("Anzahl Szenarien", 1, 8, int(defaults["scoring"]["top_n"]))
use_start = st.sidebar.checkbox("Startpunkt der Zählung festlegen")
start_date: date | None = None

expert = st.sidebar.toggle("Experten-Modus (Gewichte)")
if expert:
    st.sidebar.caption("Gewichte der Richtlinien (0 = aus)")
    weights = {}
    for key, value in defaults["guidelines"]["weights"].items():
        weights[key] = st.sidebar.slider(key, 0.0, 5.0, float(value), 0.25, key=f"w_{key}")
    overrides["guidelines"] = {"weights": weights}
    temp = st.sidebar.slider("Softmax-Temperatur", 1.0, 30.0, float(defaults["scoring"]["softmax_temperature"]), 0.5)
    overrides["scoring"] = {"softmax_temperature": temp}

# ---------------------------------------------------------------------------
# main area
# ---------------------------------------------------------------------------

st.title("Elliott-Wellen-Analyzer")
st.warning(f"⚠️ **{DISCLAIMER}**")

if error:
    st.error(f"Daten konnten nicht geladen werden: {error}")
    st.stop()
if data is None:
    st.info("Bitte links eine Datenquelle wählen (CSV hochladen, Ticker laden oder Beispieldaten nutzen).")
    st.stop()

cfg_dict = deep_merge(defaults, overrides)
settings = Settings.model_validate(cfg_dict)
if resample != "Original":
    try:
        data = resample_ohlc(data, resample, settings.data)
    except DataValidationError as exc:
        st.error(str(exc))
        st.stop()

if use_start:
    start_date = st.sidebar.date_input(
        "Startdatum", value=data.df.index[len(data.df) // 2].date(),
        min_value=data.df.index[0].date(), max_value=data.df.index[-1].date(),
    )

with st.spinner("Suche Wellenzählungen …"):
    result = _analyze(data.df, data.source, cfg_dict, start_date, log_mode, top_n)

m = result.meta
c1, c2, c3, c4 = st.columns(4)
c1.metric("Kerzen", f"{m.bars}")
c2.metric("Letzter Kurs", fmt_price(m.last_close))
c3.metric("Gültige Zählungen", f"{len(result.scenarios)}")
c4.metric("Laufzeit", f"{m.runtime_s:.2f} s")
st.caption(
    f"Quelle: {m.source} · {m.start:%Y-%m-%d} bis {m.end:%Y-%m-%d} · Skala: {'log' if m.log_scale else 'linear'} · "
    + " · ".join(f"{lv.name}: {lv.pivot_count} Pivots ({lv.threshold})" for lv in m.levels)
)
for w in [*data.warnings, *result.warnings]:
    st.info(w)

if not result.scenarios:
    st.error("Keine regelkonforme Zählung gefunden. Tipp: ZigZag-Schwelle ändern oder anderen Startpunkt wählen.")
    hierarchy = build_hierarchy(data.df, settings.pivots, settings.indicators)
    st.plotly_chart(plot_pivots(data.df, hierarchy, settings.plotting), width="stretch")
    st.stop()

top = result.scenarios[:top_n]
labels = {s.id: f"{s.id} · {s.count.pattern.german} · Score {s.score.total:.0f}" for s in top}
main_id = st.radio("Hauptszenario im Chart", list(labels), format_func=lambda k: labels[k], horizontal=True)
alts = [s.id for s in top if s.id != main_id]
shown = st.multiselect("Alternativen einblenden (auch per Klick auf die Legende)", alts, default=[])
c1, c2 = st.columns([1, 1])
log_axis = c1.toggle("Log-Achse im Chart", value=m.log_scale)
show_volume = c2.toggle("Volumen anzeigen", value=m.has_volume)

fig = plot_analysis(data.df, result, settings.plotting, scenario_ids=[main_id, *alts], log_axis=log_axis,
                    show_volume=show_volume)
for trace in fig.data:
    group = getattr(trace, "legendgroup", None)
    if group in shown:
        trace.visible = True
st.plotly_chart(fig, width="stretch")

st.subheader("Szenarien")
st.caption(f"ℹ️ {PROBABILITY_NOTE}")
rows = []
for s in top:
    p = s.projection
    rows.append({
        "Rang": s.rank,
        "Muster": s.count.pattern.german,
        "Aktuelle Position": p.position if p else "–",
        "Score": round(s.score.total, 1),
        "rel. Gewichtung (heuristisch)": f"{s.probability * 100:.0f} %" if s.probability is not None else "–",
        "Ziele": "; ".join(f"{fmt_price(z.low)}–{fmt_price(z.high)}{' ★' if z.highlighted else ''}" for z in p.targets) if p else "–",
        "Invalidierung": f"{'<' if p.invalidation.side == 'below' else '>'} {fmt_price(p.invalidation.price)}" if p else "–",
    })
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
st.caption("★ = Konfluenz (mehrere Fibonacci-Verhältnisse überlagern sich)")

for s in top:
    with st.expander(f"{s.id}: {s.count.pattern.german} – {s.projection.position if s.projection else ''}", expanded=s.id == main_id):
        st.markdown(scenario_markdown(s))

st.subheader("Export")
c1, c2, c3 = st.columns(3)
c1.download_button("JSON", to_json(result), file_name="elliott_analyse.json", mime="application/json")
c2.download_button("Markdown", to_markdown(result, top=top_n), file_name="elliott_analyse.md", mime="text/markdown")
html_fig = fig.to_html(full_html=False, include_plotlyjs="cdn")
c3.download_button("HTML (mit Chart)", to_html(result, html_fig, top=top_n), file_name="elliott_analyse.html", mime="text/html")

with st.expander("Pivot-Kontrolle (ZigZag je Grad)"):
    hierarchy = build_hierarchy(data.df, settings.pivots, settings.indicators)
    st.plotly_chart(plot_pivots(data.df, hierarchy, settings.plotting, log_axis=log_axis), width="stretch")

st.divider()
st.caption(f"⚠️ {DISCLAIMER}")
