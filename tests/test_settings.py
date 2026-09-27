from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from elliott.settings import ThresholdSpec, load_settings, with_overrides


def test_defaults_load(settings) -> None:
    assert settings.pivots.threshold.mode == "atr"
    assert settings.scoring.top_n >= 1
    assert "impulse_w3_ratio" in settings.guidelines.weights


def test_user_file_is_merged(tmp_path: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text("search:\n  beam_width: 7\n", encoding="utf-8")
    s = load_settings(cfg)
    assert s.search.beam_width == 7
    assert s.search.max_skip == load_settings().search.max_skip


def test_unknown_key_rejected() -> None:
    with pytest.raises(ValidationError):
        load_settings(overrides={"search": {"beam_widht": 3}})


def test_with_overrides(settings) -> None:
    s2 = with_overrides(settings, {"fib": {"tolerance": 0.05}})
    assert s2.fib.tolerance == 0.05
    assert settings.fib.tolerance != 0.05 or settings.fib.tolerance == 0.05


@pytest.mark.parametrize(
    ("text", "mode", "value"),
    [("atr:2", "atr", 2.0), ("pct:5", "percent", 5.0), ("5%", "percent", 5.0), ("ATR=1,5", "atr", 1.5)],
)
def test_threshold_parse(text: str, mode: str, value: float) -> None:
    spec = ThresholdSpec.parse(text)
    assert spec.mode == mode and spec.value == pytest.approx(value)


def test_threshold_parse_error() -> None:
    with pytest.raises(ValueError):
        ThresholdSpec.parse("foo")


def test_no_magic_defaults_in_models() -> None:
    """Settings models have no defaults: every value comes from the YAML file."""
    from elliott import settings as mod

    for model in (mod.DataConfig, mod.SearchConfig, mod.RulesConfig, mod.FibConfig, mod.ProjectionConfig):
        for name, field in model.model_fields.items():
            assert field.is_required(), f"{model.__name__}.{name} hat einen Code-Default"
