"""Smoke test of the Streamlit app (headless)."""

from __future__ import annotations

from pathlib import Path

import pytest

APP = str(Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py")


@pytest.fixture
def app_test():
    from streamlit.testing.v1 import AppTest

    return AppTest.from_file(APP, default_timeout=60)


def test_app_runs_with_example_data(app_test) -> None:
    at = app_test.run()
    assert not at.exception, at.exception
    warnings = " ".join(w.value for w in at.warning)
    assert "Keine Anlageberatung" in warnings
    assert len(at.dataframe) >= 1
    assert any("Szenarien" in s.value for s in at.subheader)
    captions = " ".join(c.value for c in at.caption)
    assert "keine statistische Wahrscheinlichkeit" in captions


def test_app_expert_mode(app_test) -> None:
    at = app_test.run()
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception, at.exception
    assert len(at.sidebar.slider) > 5
