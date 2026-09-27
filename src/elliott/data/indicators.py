"""Technical indicators (own implementations, no TA-Lib dependency)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def true_range(df: pd.DataFrame) -> pd.Series:
    """True range of each bar."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    """Average True Range (Wilder smoothing).

    The warm-up phase uses an expanding mean so that a value exists from the
    first bar on (the ZigZag needs a threshold everywhere).
    """
    tr = true_range(df)
    wilder = tr.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    warmup = tr.expanding(min_periods=1).mean()
    return wilder.fillna(warmup)


def rsi(close: pd.Series, period: int) -> pd.Series:
    """Relative Strength Index (Wilder). Warm-up values are neutral (50)."""
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(avg_loss != 0.0, 100.0)
    return out.fillna(50.0)
