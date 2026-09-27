"""Price data import: CSV (many dialects), yfinance, validation, resampling.

The public entry points return a :class:`MarketData` object with a clean
DataFrame (``DatetimeIndex``, float columns ``open, high, low, close`` and an
optional ``volume``) plus a list of human readable warnings (German).
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO

import numpy as np
import pandas as pd

from elliott.settings import DataConfig

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close")

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": (
        "date", "datum", "time", "timestamp", "zeit", "datetime", "day", "tag",
        "handelstag", "zeitstempel", "date time", "open time", "unix", "unix timestamp",
    ),
    "open": (
        "open", "eröffnung", "eroeffnung", "eröffnungskurs", "eroeffnungskurs", "erster",
        "erster kurs", "öffnung", "oeffnung", "eröffnungspreis", "start", "o",
    ),
    "high": ("high", "hoch", "höchst", "hoechst", "höchstkurs", "hoechstkurs", "tageshoch", "max", "maximum", "h"),
    "low": ("low", "tief", "tiefst", "tiefstkurs", "tagestief", "min", "minimum", "l"),
    "close": (
        "close", "schluss", "schlusskurs", "letzter", "letzter kurs", "schlusspreis", "last",
        "price", "kurs", "c",
    ),
    "adj_close": ("adj close", "adj. close", "adjclose", "adjusted close", "bereinigter schluss"),
    "volume": ("volume", "vol", "vol.", "volumen", "umsatz", "stück", "stueck", "anzahl", "v"),
}

YF_INTERVAL_TO_PANDAS: dict[str, str] = {
    "1m": "1min", "2m": "2min", "5m": "5min", "15m": "15min", "30m": "30min",
    "60m": "1h", "90m": "90min", "1h": "1h", "2h": "2h", "4h": "4h",
    "1d": "1D", "d": "1D", "5d": "5D", "1wk": "W", "1w": "W", "w": "W",
    "1mo": "MS", "mo": "MS", "m": "MS", "3mo": "QS",
}

_TZ_SUFFIX = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")
_DOTTED_DATE = re.compile(r"^\d{1,2}\.\d{1,2}\.\d{2,4}")


class DataValidationError(ValueError):
    """Raised when price data cannot be used; the message is user facing (German)."""


@dataclass
class MarketData:
    """Validated OHLC(V) data plus metadata."""

    df: pd.DataFrame
    source: str
    warnings: list[str] = field(default_factory=list)

    @property
    def has_volume(self) -> bool:
        return "volume" in self.df.columns and bool(self.df["volume"].notna().any())

    @property
    def price_ratio(self) -> float:
        return float(self.df["high"].max() / self.df["low"].min())


# ---------------------------------------------------------------------------
# CSV parsing helpers
# ---------------------------------------------------------------------------


def _normalize_name(name: object) -> str:
    text = str(name).strip().strip('"').strip("'").lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _detect_separator(sample: str) -> str:
    header = next((line for line in sample.splitlines() if line.strip()), "")
    counts = {sep: header.count(sep) for sep in (";", "\t", "|", ",")}
    best = max(counts, key=lambda s: counts[s])
    if counts[best] == 0:
        return ","
    # Semicolon wins ties: German exports use ';' plus decimal commas.
    if counts[";"] > 0 and counts[";"] >= counts[","] - 1:
        return ";"
    return best


def parse_number_series(series: pd.Series, decimal_comma_hint: bool = False) -> pd.Series:
    """Convert a column of numbers in any common notation to float.

    Handles decimal commas (``1.234,56``), thousands separators (``1,234.56``),
    spaces/apostrophes as grouping and stray whitespace.
    """
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)
    text = series.astype("string").str.strip()
    text = text.str.replace(r"[\s '’]", "", regex=True)
    text = text.replace({"": pd.NA, "-": pd.NA, "null": pd.NA, "None": pd.NA, "nan": pd.NA, "NaN": pd.NA})
    non_null = text.dropna()
    has_comma = bool(non_null.str.contains(",", regex=False).any())
    has_dot = bool(non_null.str.contains(".", regex=False).any())
    if has_comma and has_dot:
        last_comma = non_null.str.rfind(",")
        last_dot = non_null.str.rfind(".")
        both = (last_comma >= 0) & (last_dot >= 0)
        comma_decimal = bool((last_comma[both] > last_dot[both]).mean() > 0.5) if both.any() else decimal_comma_hint
        if comma_decimal:
            text = text.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
        else:
            text = text.str.replace(",", "", regex=False)
    elif has_dot and decimal_comma_hint:
        # German export: dots can only be thousands separators ("1.200").
        if bool(non_null.str.fullmatch(r"-?\d{1,3}(\.\d{3})+|-?\d+").all()):
            text = text.str.replace(".", "", regex=False)
    elif has_comma:
        thousands_like = bool(non_null.str.fullmatch(r"-?\d{1,3}(,\d{3})+").all())
        if thousands_like and not decimal_comma_hint:
            text = text.str.replace(",", "", regex=False)
        else:
            text = text.str.replace(",", ".", regex=False)
    return pd.to_numeric(text, errors="coerce").astype(float)


def parse_dates(series: pd.Series, cfg: DataConfig) -> pd.Series:
    """Parse dates in ISO, German (``31.12.2023``), US or Unix-timestamp notation.

    Time zone suffixes are dropped so that the exchange's local wall time is
    kept (important for daily bars exported with an offset, e.g. TradingView).
    """
    if pd.api.types.is_datetime64_any_dtype(series):
        values = series.dt.tz_localize(None) if getattr(series.dt, "tz", None) is not None else series
        return values.astype("datetime64[ns]")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() >= cfg.date_parse_min_success and float(numeric.min()) >= cfg.unix_min_value:
        unit = "ms" if float(numeric.median()) > cfg.unix_ms_threshold else "s"
        return pd.Series(pd.to_datetime(numeric, unit=unit), index=series.index)

    text = series.astype("string").str.strip()
    text = text.str.replace(_TZ_SUFFIX, "", regex=True).str.replace("T", " ", regex=False)
    non_null = text.dropna()
    dotted = bool(non_null.str.match(_DOTTED_DATE).mean() > 0.5) if len(non_null) else False

    candidates: list[tuple[str, pd.Series]] = []
    iso = pd.to_datetime(text, errors="coerce", format="ISO8601")
    candidates.append(("iso", iso))
    day_first = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=True)
    month_first = pd.to_datetime(text, errors="coerce", format="mixed", dayfirst=False)
    if dotted:
        candidates.insert(0, ("dayfirst", day_first))
    else:
        candidates.append(("dayfirst", day_first))
    candidates.append(("monthfirst", month_first))

    def quality(parsed: pd.Series) -> tuple[float, float]:
        ok = parsed.notna().mean()
        valid = parsed.dropna()
        monotone = float((valid.diff().dropna() >= pd.Timedelta(0)).mean()) if len(valid) > 1 else 1.0
        # Accept either ascending or descending files.
        monotone = max(monotone, 1.0 - monotone)
        return float(ok), monotone

    best_name, best = candidates[0]
    best_q = quality(best)
    for name, parsed in candidates[1:]:
        q = quality(parsed)
        if q > best_q:
            best_name, best, best_q = name, parsed, q
    if best_q[0] < cfg.date_parse_min_success:
        examples = ", ".join(map(str, series.dropna().astype(str).head(3).tolist()))
        raise DataValidationError(
            f"Datumsspalte konnte nicht gelesen werden (nur {best_q[0]:.0%} gültige Werte). "
            f"Beispiele: {examples}"
        )
    return best


def _map_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names -> original column names."""
    normalized = {col: _normalize_name(col) for col in df.columns}
    mapping: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            match = next((col for col, norm in normalized.items() if norm == alias and col not in mapping.values()), None)
            if match is not None:
                mapping[canonical] = match
                break
    if "date" not in mapping:
        # Fallback: common prefixes such as "Date (UTC)" or "Zeit/Datum".
        for col, norm in normalized.items():
            if any(norm.startswith(p) for p in ("date", "datum", "time", "zeit")) and col not in mapping.values():
                mapping["date"] = col
                break
    return mapping


def read_csv_any(source: str | Path | bytes | IO[str] | IO[bytes], cfg: DataConfig) -> MarketData:
    """Read a CSV export (TradingView, Yahoo, German broker exports, ...)."""
    if isinstance(source, (str, Path)) and Path(str(source)).exists():
        raw_bytes = Path(source).read_bytes()
        name = Path(source).name
    elif isinstance(source, bytes):
        raw_bytes, name = source, "upload"
    elif hasattr(source, "read"):
        content = source.read()
        raw_bytes = content if isinstance(content, bytes) else content.encode("utf-8")
        name = getattr(source, "name", "upload")
    else:
        raise DataValidationError(f"Datei nicht gefunden: {source}")

    text = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None or not text.strip():
        raise DataValidationError("Die CSV-Datei ist leer oder nicht lesbar.")

    sep = _detect_separator(text[:4096])
    try:
        frame = pd.read_csv(io.StringIO(text), sep=sep, dtype=str, skipinitialspace=True)
    except Exception as exc:  # pragma: no cover - pandas parser errors vary
        raise DataValidationError(f"CSV konnte nicht gelesen werden: {exc}") from exc
    frame = frame.dropna(axis=1, how="all")
    return frame_to_market_data(frame, cfg, source=name, decimal_comma_hint=(sep == ";"))


def frame_to_market_data(
    frame: pd.DataFrame,
    cfg: DataConfig,
    source: str = "DataFrame",
    decimal_comma_hint: bool = False,
) -> MarketData:
    """Normalize an arbitrary OHLC DataFrame into validated :class:`MarketData`."""
    frame = frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex):
        frame = frame.reset_index(names="date")
    mapping = _map_columns(frame)
    if "close" not in mapping and "adj_close" in mapping:
        mapping["close"] = mapping["adj_close"]
    missing = [c for c in ("date", *PRICE_COLUMNS) if c not in mapping]
    if missing:
        german = {"date": "Datum", "open": "Open", "high": "High", "low": "Low", "close": "Close"}
        raise DataValidationError(
            "Pflichtspalten fehlen: "
            + ", ".join(german[m] for m in missing)
            + f". Gefundene Spalten: {', '.join(map(str, frame.columns))}"
        )

    out = pd.DataFrame(index=frame.index)
    out["date"] = parse_dates(frame[mapping["date"]], cfg)
    for col in PRICE_COLUMNS:
        out[col] = parse_number_series(frame[mapping[col]], decimal_comma_hint)
    for col in PRICE_COLUMNS:
        raw = frame[mapping[col]].astype("string").str.strip()
        present = raw.notna() & (raw != "")
        ratio = float(out.loc[present, col].notna().mean()) if present.any() else 0.0
        if ratio < cfg.column_detect_min_numeric:
            raise DataValidationError(
                f"Spalte '{mapping[col]}' enthält zu wenige Zahlen ({ratio:.0%}). "
                "Bitte Dezimaltrennzeichen und Spaltenzuordnung prüfen."
            )
    if "volume" in mapping:
        out["volume"] = parse_number_series(frame[mapping["volume"]], decimal_comma_hint)
    out = out.dropna(subset=["date"]).set_index("date")
    out.index.name = "date"
    return validate_ohlc(out, cfg, source=source)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_ohlc(df: pd.DataFrame, cfg: DataConfig, source: str = "data") -> MarketData:
    """Sort, de-duplicate, fill gaps and check OHLC consistency.

    Raises :class:`DataValidationError` with a German message on fatal problems.
    """
    warnings: list[str] = []
    df = df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataValidationError("Der Index muss aus Zeitstempeln bestehen.")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    keep = [c for c in (*PRICE_COLUMNS, "volume") if c in df.columns]
    df = df[keep].astype(float)

    if not df.index.is_monotonic_increasing:
        df = df.sort_index()
        warnings.append("Daten waren nicht chronologisch sortiert und wurden sortiert.")
    dup = int(df.index.duplicated(keep="last").sum())
    if dup:
        df = df[~df.index.duplicated(keep="last")]
        warnings.append(f"{dup} doppelte Zeitstempel entfernt (jeweils letzter Eintrag behalten).")

    all_nan = df[list(PRICE_COLUMNS)].isna().all(axis=1)
    if all_nan.any():
        df = df[~all_nan]
        warnings.append(f"{int(all_nan.sum())} Zeilen ohne Kursdaten entfernt.")
    if len(df) == 0:
        raise DataValidationError("Keine gültigen Kursdaten gefunden.")

    partial = df[list(PRICE_COLUMNS)].isna().any(axis=1)
    if partial.mean() > cfg.max_nan_ratio:
        raise DataValidationError(
            f"{partial.mean():.0%} der Zeilen sind unvollständig (Grenze {cfg.max_nan_ratio:.0%})."
        )
    if partial.any():
        close = df["close"].ffill().bfill()
        open_ = df["open"].fillna(close.shift(1)).fillna(close)
        close = close.where(df["close"].notna(), open_)
        body_max = np.maximum(open_, close)
        body_min = np.minimum(open_, close)
        df["close"] = close
        df["open"] = open_
        df["high"] = df["high"].fillna(body_max)
        df["low"] = df["low"].fillna(body_min)
        warnings.append(f"{int(partial.sum())} unvollständige Zeilen ergänzt (Vorwert bzw. Kerzenkörper).")

    if (df[list(PRICE_COLUMNS)] <= 0).any().any():
        bad = df.index[(df[list(PRICE_COLUMNS)] <= 0).any(axis=1)][:3]
        raise DataValidationError(
            "Kurse müssen größer als 0 sein. Betroffene Zeitpunkte: " + ", ".join(map(str, bad))
        )

    body_max = df[["open", "close"]].max(axis=1)
    body_min = df[["open", "close"]].min(axis=1)
    high_viol = (body_max - df["high"]).clip(lower=0) / body_max
    low_viol = (df["low"] - body_min).clip(lower=0) / body_min
    swap = df["high"] < df["low"]
    violation = (high_viol > 0) | (low_viol > 0) | swap
    if violation.any():
        worst = float(max(high_viol.max(), low_viol.max()))
        count = int(violation.sum())
        if not cfg.ohlc_repair or worst > cfg.ohlc_repair_max_rel:
            examples = ", ".join(map(str, df.index[violation][:3]))
            raise DataValidationError(
                f"{count} Kerzen verletzen High >= max(Open, Close) bzw. Low <= min(Open, Close) "
                f"(größte Abweichung {worst:.2%}). Beispiele: {examples}"
            )
        hi = df[list(PRICE_COLUMNS)].max(axis=1)
        lo = df[list(PRICE_COLUMNS)].min(axis=1)
        df["high"], df["low"] = hi, lo
        warnings.append(f"{count} Kerzen mit leicht inkonsistentem High/Low korrigiert (max. {worst:.2%}).")

    if "volume" in df.columns:
        if df["volume"].notna().sum() == 0 or float(df["volume"].fillna(0).sum()) == 0.0:
            df = df.drop(columns="volume")
            warnings.append("Keine Volumendaten vorhanden – Volumen-Richtlinien werden neutral bewertet.")
        else:
            df["volume"] = df["volume"].fillna(0.0).clip(lower=0.0)

    if len(df) < cfg.min_bars:
        raise DataValidationError(
            f"Zu wenige Kerzen: {len(df)} (mindestens {cfg.min_bars} nötig für eine Wellenanalyse)."
        )

    if len(df) > 2:
        diffs = df.index.to_series().diff().dropna()
        median = diffs.median()
        if median > pd.Timedelta(0):
            gaps = diffs[diffs > median * cfg.gap_warning_factor]
            if len(gaps):
                warnings.append(
                    f"{len(gaps)} größere Datenlücken erkannt (größte: {gaps.max()} ab {gaps.idxmax()})."
                )
    return MarketData(df=df, source=source, warnings=warnings)


# ---------------------------------------------------------------------------
# Resampling, log scale, yfinance
# ---------------------------------------------------------------------------


def resample_ohlc(data: MarketData, rule: str, cfg: DataConfig) -> MarketData:
    """Aggregate to a higher timeframe (e.g. ``1wk``, ``W``, ``1mo``, ``4h``)."""
    pandas_rule = YF_INTERVAL_TO_PANDAS.get(rule.lower(), rule)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in data.df.columns:
        agg["volume"] = "sum"
    try:
        out = data.df.resample(pandas_rule).agg(agg).dropna(subset=["open", "high", "low", "close"])
    except (ValueError, TypeError) as exc:
        raise DataValidationError(f"Unbekannte Zeiteinheit für Resampling: '{rule}'") from exc
    result = validate_ohlc(out, cfg, source=f"{data.source} ({rule})")
    result.warnings = [*data.warnings, *result.warnings]
    return result


def resolve_log_scale(df: pd.DataFrame, mode: str, auto_ratio: float) -> bool:
    """Decide whether lengths and Fibonacci ratios are measured in log prices."""
    if mode == "log":
        return True
    if mode == "linear":
        return False
    return float(df["high"].max() / df["low"].min()) > auto_ratio


def load_ticker(
    ticker: str,
    cfg: DataConfig,
    period: str | None = "2y",
    interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> MarketData:
    """Download data via yfinance (requires network access)."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise DataValidationError("yfinance ist nicht installiert.") from exc
    kwargs: dict[str, object] = {"interval": interval, "auto_adjust": False}
    if start or end:
        kwargs.update({"start": start, "end": end})
    else:
        kwargs["period"] = period
    try:
        frame = yf.Ticker(ticker).history(**kwargs)
    except Exception as exc:  # pragma: no cover - network errors vary
        raise DataValidationError(f"Download für '{ticker}' fehlgeschlagen: {exc}") from exc
    if frame is None or frame.empty:
        raise DataValidationError(
            f"Keine Daten für Ticker '{ticker}' (Zeitraum {period or f'{start}–{end}'}, Intervall {interval})."
        )
    frame = frame.copy()
    if isinstance(frame.index, pd.DatetimeIndex) and frame.index.tz is not None:
        frame.index = frame.index.tz_localize(None)
    frame.index.name = "date"
    return frame_to_market_data(frame, cfg, source=f"yfinance:{ticker}:{interval}")


def load_data(
    source: str | Path | bytes | IO[str] | IO[bytes] | pd.DataFrame,
    cfg: DataConfig,
) -> MarketData:
    """Convenience loader for CSV paths/bytes/file objects or DataFrames."""
    if isinstance(source, pd.DataFrame):
        return frame_to_market_data(source, cfg)
    return read_csv_any(source, cfg)
