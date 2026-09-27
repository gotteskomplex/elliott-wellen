"""Regenerate the example CSV files (synthetic data with known structure).

Run from the repository root: ``python examples/make_examples.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tests.synthetic import make_pattern  # noqa: E402

OUT = Path(__file__).resolve().parent


def main() -> None:
    # 1) Textbook impulse, then the start of a correction – TradingView export format.
    syn = make_pattern("impulse", bars=420, start_price=100.0, height=60.0, noise=0.006, seed=42, tail=25)
    df = syn.df.copy()
    tv = df.rename(columns={"volume": "Volume"})
    tv.insert(0, "time", ((df.index - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).astype(np.int64))
    tv.to_csv(OUT / "beispiel_impuls_tradingview.csv", index=False, float_format="%.4f")

    # 2) Zigzag correction after an impulsive decline – German broker format (; and decimal comma).
    syn2 = make_pattern("zigzag", bars=260, start_price=4000.0, height=900.0, noise=0.004, seed=7, prefix=2.0, direction=1)
    de = syn2.df.copy()
    de.index = de.index.strftime("%d.%m.%Y")
    de = de.rename(columns={"open": "Eröffnung", "high": "Hoch", "low": "Tief", "close": "Schlusskurs", "volume": "Volumen"})
    de.to_csv(OUT / "beispiel_zigzag_deutsch.csv", sep=";", decimal=",", index_label="Datum", float_format="%.2f", encoding="utf-8")
    print("Beispieldaten geschrieben nach", OUT)


if __name__ == "__main__":
    main()
