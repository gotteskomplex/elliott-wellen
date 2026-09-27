from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from elliott.data.indicators import atr, rsi
from elliott.data.loader import (
    DataValidationError,
    frame_to_market_data,
    load_data,
    parse_number_series,
    read_csv_any,
    resample_ohlc,
    resolve_log_scale,
)

TRADINGVIEW = """time,open,high,low,close,Volume
1672617600,100,105,99,104,1000
1672704000,104,106,101,102,1200
1672790400,102,103,98,99,900
"""

YAHOO = """Date,Open,High,Low,Close,Adj Close,Volume
2023-01-03,100.0,105.0,99.0,104.0,103.5,1000
2023-01-04,104.0,106.0,101.0,102.0,101.5,1200
2023-01-05,102.0,103.0,98.0,99.0,98.5,900
"""

GERMAN = """Datum;Eröffnung;Hoch;Tief;Schlusskurs;Volumen
03.01.2023;1.100,50;1.105,00;1.099,00;1.104,25;1.000
04.01.2023;1.104,25;1.106,00;1.101,00;1.102,00;1.200
05.01.2023;1.102,00;1.103,00;1.098,00;1.099,00;900
"""


def _cfg(settings, **kw):
    return settings.data.model_copy(update={"min_bars": 2, **kw})


def test_tradingview_unix(settings) -> None:
    md = read_csv_any(TRADINGVIEW.encode(), _cfg(settings))
    assert list(md.df.columns) == ["open", "high", "low", "close", "volume"]
    assert md.df.index[0] == pd.Timestamp("2023-01-02")
    assert md.has_volume


def test_tradingview_iso_with_offset(settings) -> None:
    text = "time,open,high,low,close\n2023-01-02T00:00:00+01:00,1,2,0.5,1.5\n2023-01-03T00:00:00+01:00,1.5,2,1,1.2\n"
    md = read_csv_any(text.encode(), _cfg(settings))
    assert md.df.index[0] == pd.Timestamp("2023-01-02")  # local wall time kept
    assert not md.has_volume


def test_yahoo(settings) -> None:
    md = read_csv_any(YAHOO.encode(), _cfg(settings))
    assert md.df["close"].iloc[0] == 104.0  # 'Close' preferred over 'Adj Close'


def test_german_format(settings) -> None:
    md = read_csv_any(GERMAN.encode("cp1252"), _cfg(settings))
    assert md.df.index[0] == pd.Timestamp("2023-01-03")
    assert md.df["open"].iloc[0] == pytest.approx(1100.5)
    assert md.df["close"].iloc[0] == pytest.approx(1104.25)
    assert md.df["volume"].iloc[1] == pytest.approx(1200)


def test_file_object_and_path(settings, tmp_path) -> None:
    path = tmp_path / "x.csv"
    path.write_text(YAHOO, encoding="utf-8")
    assert len(load_data(path, _cfg(settings)).df) == 3
    assert len(load_data(io.StringIO(YAHOO), _cfg(settings)).df) == 3


def test_sort_dedupe_and_warnings(settings) -> None:
    text = YAHOO.splitlines()
    shuffled = "\n".join([text[0], text[3], text[1], text[2], text[2]]) + "\n"
    md = read_csv_any(shuffled.encode(), _cfg(settings))
    assert md.df.index.is_monotonic_increasing
    assert len(md.df) == 3
    assert any("sortiert" in w for w in md.warnings)
    assert any("doppelte" in w for w in md.warnings)


def test_missing_columns(settings) -> None:
    with pytest.raises(DataValidationError, match="Pflichtspalten fehlen"):
        read_csv_any(b"Date,Open,Close\n2023-01-01,1,2\n", _cfg(settings))


def test_ohlc_small_violation_repaired(settings) -> None:
    text = "Date,Open,High,Low,Close\n2023-01-02,100,100.5,99,101\n2023-01-03,101,102,100,101.5\n"
    md = read_csv_any(text.encode(), _cfg(settings))
    assert md.df["high"].iloc[0] == pytest.approx(101)
    assert any("korrigiert" in w for w in md.warnings)


def test_ohlc_large_violation_error(settings) -> None:
    text = "Date,Open,High,Low,Close\n2023-01-02,100,90,80,101\n2023-01-03,101,102,100,101.5\n"
    with pytest.raises(DataValidationError, match="verletzen"):
        read_csv_any(text.encode(), _cfg(settings))


def test_nan_handling(settings) -> None:
    text = "Date,Open,High,Low,Close\n2023-01-02,100,102,99,101\n2023-01-03,,,,\n2023-01-04,101,,100,101.5\n2023-01-05,101,103,100,102\n"
    md = read_csv_any(text.encode(), _cfg(settings, max_nan_ratio=0.5))
    assert len(md.df) == 3
    assert md.df["high"].iloc[1] == pytest.approx(101.5)


def test_non_positive_prices(settings) -> None:
    text = "Date,Open,High,Low,Close\n2023-01-02,0,1,0,1\n2023-01-03,1,2,1,2\n"
    with pytest.raises(DataValidationError, match="größer als 0"):
        read_csv_any(text.encode(), _cfg(settings))


def test_too_few_bars(settings) -> None:
    with pytest.raises(DataValidationError, match="Zu wenige Kerzen"):
        read_csv_any(YAHOO.encode(), settings.data)


def test_bad_dates(settings) -> None:
    text = "Date,Open,High,Low,Close\nfoo,1,2,1,2\nbar,1,2,1,2\n"
    with pytest.raises(DataValidationError, match="Datumsspalte"):
        read_csv_any(text.encode(), _cfg(settings))


@pytest.mark.parametrize(
    ("values", "hint", "expected"),
    [
        (["1.234,56", "2,5"], False, [1234.56, 2.5]),
        (["1,234.56", "2.5"], False, [1234.56, 2.5]),
        (["1,234", "2,500"], False, [1234.0, 2500.0]),
        (["1,234", "2,500"], True, [1.234, 2.5]),
        (["12,5", "3,75"], False, [12.5, 3.75]),
        (["1 234.5", "7"], False, [1234.5, 7.0]),
    ],
)
def test_parse_numbers(values, hint, expected) -> None:
    out = parse_number_series(pd.Series(values), decimal_comma_hint=hint)
    assert out.tolist() == pytest.approx(expected)


def _daily(n: int = 30) -> pd.DataFrame:
    idx = pd.date_range("2023-01-02", periods=n, freq="D")
    close = np.linspace(100, 130, n)
    return pd.DataFrame(
        {"open": close - 0.5, "high": close + 1, "low": close - 1, "close": close, "volume": 100.0},
        index=idx,
    )


def test_resample_weekly(settings) -> None:
    md = frame_to_market_data(_daily(), _cfg(settings))
    wk = resample_ohlc(md, "1wk", _cfg(settings))
    assert len(wk.df) < len(md.df)
    first_week = md.df.loc[: wk.df.index[0]]
    assert wk.df["high"].iloc[0] == pytest.approx(first_week["high"].max())
    assert wk.df["volume"].iloc[0] == pytest.approx(first_week["volume"].sum())


def test_log_scale_decision() -> None:
    df = pd.DataFrame({"high": [10.0, 40.0], "low": [9.0, 12.0]})
    assert resolve_log_scale(df, "auto", 3.0) is True
    assert resolve_log_scale(df, "linear", 3.0) is False
    df2 = pd.DataFrame({"high": [10.0, 20.0], "low": [9.0, 12.0]})
    assert resolve_log_scale(df2, "auto", 3.0) is False
    assert resolve_log_scale(df2, "log", 3.0) is True


def test_indicators() -> None:
    df = _daily(40)
    a = atr(df, 14)
    assert a.notna().all() and (a > 0).all()
    r = rsi(df["close"], 14)
    assert r.notna().all()
    assert r.iloc[-1] > 90  # monotone rising close


def test_missing_volume_warning(settings) -> None:
    df = _daily().assign(volume=0.0)
    md = frame_to_market_data(df, _cfg(settings))
    assert not md.has_volume
    assert any("Volumen" in w for w in md.warnings)
