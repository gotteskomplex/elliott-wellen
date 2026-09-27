"""Configuration models and YAML loading.

All thresholds, tolerances, weights and Fibonacci ratios live in
``elliott/config/default.yaml``. The Pydantic models below intentionally carry
*no* default values: every value must come from the YAML file, so there are no
magic numbers hidden in the code. User configurations only contain the values
that differ and are deep-merged over the defaults.
"""

from __future__ import annotations

import copy
import re
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

DEFAULT_CONFIG_RESOURCE = "config/default.yaml"


class _Strict(BaseModel):
    """Base model: unknown keys are errors (catches typos in user configs)."""

    model_config = ConfigDict(extra="forbid", frozen=False)


class ThresholdSpec(_Strict):
    """A ZigZag reversal threshold, either in percent or as ATR multiple."""

    mode: Literal["percent", "atr"]
    value: float = Field(gt=0)

    @classmethod
    def parse(cls, text: str) -> "ThresholdSpec":
        """Parse CLI notation such as ``atr:2``, ``pct:5``, ``percent:5`` or ``5%``."""
        raw = text.strip().lower().replace(",", ".")
        match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*%", raw)
        if match:
            return cls(mode="percent", value=float(match.group(1)))
        match = re.fullmatch(r"(atr|pct|percent|prozent)\s*[:=]\s*(\d+(?:\.\d+)?)", raw)
        if not match:
            raise ValueError(
                f"Ungültige Schwelle '{text}'. Erwartet z. B. 'atr:2', 'pct:5' oder '5%'."
            )
        mode = "atr" if match.group(1) == "atr" else "percent"
        return cls(mode=mode, value=float(match.group(2)))

    def scaled(self, factor: float) -> "ThresholdSpec":
        """Return a copy with the value multiplied by ``factor``."""
        return ThresholdSpec(mode=self.mode, value=self.value * factor)

    def describe(self) -> str:
        """Human readable description (German)."""
        if self.mode == "atr":
            return f"{self.value:g} × ATR"
        return f"{self.value:g} %"


class DataConfig(_Strict):
    min_bars: int = Field(ge=2)
    log_mode: Literal["auto", "linear", "log"]
    log_auto_ratio: float = Field(gt=1)
    ohlc_repair: bool
    ohlc_repair_max_rel: float = Field(ge=0)
    max_nan_ratio: float = Field(ge=0, le=1)
    gap_warning_factor: float = Field(gt=1)
    unix_ms_threshold: float
    unix_min_value: float
    date_parse_min_success: float = Field(gt=0, le=1)
    column_detect_min_numeric: float = Field(gt=0, le=1)


class IndicatorConfig(_Strict):
    atr_period: int = Field(ge=1)
    rsi_period: int = Field(ge=1)


class PivotConfig(_Strict):
    threshold: ThresholdSpec
    degree_multipliers: list[float]
    degree_names: list[str]

    @field_validator("degree_multipliers")
    @classmethod
    def _positive_sorted(cls, v: list[float]) -> list[float]:
        if not v or any(m <= 0 for m in v):
            raise ValueError("degree_multipliers müssen positiv sein")
        if sorted(v, reverse=True) != v:
            raise ValueError("degree_multipliers müssen von grob (groß) nach fein (klein) sortiert sein")
        return v


class SearchConfig(_Strict):
    level: int = Field(ge=0)
    max_pivots: int = Field(ge=4)
    min_pivots: int = Field(ge=2)
    max_skip: int = Field(ge=0)
    beam_width: int = Field(ge=1)
    max_start_candidates: int = Field(ge=1)
    max_start_lookback: int = Field(ge=2)
    min_waves: int = Field(ge=1)
    rescore_top: int = Field(ge=1)
    max_scenarios: int = Field(ge=1)
    subwave_depth: int = Field(ge=0)
    subwave_max_skip: int = Field(ge=0)
    subwave_beam_width: int = Field(ge=1)
    subwave_max_inner_pivots: int = Field(ge=4)
    subwave_min_inner_motive: int = Field(ge=2)
    subwave_min_inner_corrective: int = Field(ge=1)
    subwave_validated_base: float = Field(ge=0, le=1)
    subwave_fit_weight: float = Field(ge=0, le=1)
    irregular_start_max_excess: float = Field(ge=0)
    dedup_bar_tol: int = Field(ge=0)
    dedup_price_tol: float = Field(ge=0)
    time_budget_s: float = Field(gt=0)


class RulesConfig(_Strict):
    equality_tol: float = Field(ge=0)
    flat_b_min_retrace: float = Field(gt=0)
    flat_b_max_retrace: float = Field(gt=0)
    flat_regular_b_max: float = Field(gt=0)
    flat_expanded_b_min: float = Field(gt=0)
    flat_regular_c_shortfall: float = Field(ge=0)
    flat_running_c_min: float = Field(ge=0)
    triangle_line_tol: float = Field(ge=0)
    diagonal_require_overlap: bool
    diagonal_trendline_tol: float = Field(ge=0)
    combination_x_max: float = Field(gt=0)
    combination_max_overshoot: float = Field(ge=0)
    combination_max_triangles: int = Field(ge=0)


class FibConfig(_Strict):
    tolerance: float = Field(ge=0)
    decay_sigma: float = Field(gt=0)


class ComplexityConfig(_Strict):
    weight: float = Field(ge=0)
    by_pattern: dict[str, float]


class GuidelineConfig(_Strict):
    weights: dict[str, float]
    params: dict[str, Any]
    penalties: dict[str, float]
    complexity: ComplexityConfig

    def p(self, key: str) -> Any:
        """Return a guideline parameter, with a clear error for missing keys."""
        try:
            return self.params[key]
        except KeyError as exc:  # pragma: no cover - config error path
            raise KeyError(f"Guideline-Parameter '{key}' fehlt in der Konfiguration") from exc

    def w(self, key: str) -> float:
        """Return a guideline weight (missing weight -> config error)."""
        try:
            return float(self.weights[key])
        except KeyError as exc:  # pragma: no cover - config error path
            raise KeyError(f"Guideline-Gewicht '{key}' fehlt in der Konfiguration") from exc


class ScoringConfig(_Strict):
    softmax_temperature: float = Field(gt=0)
    top_n: int = Field(ge=1)


class ProjectionConfig(_Strict):
    targets: dict[str, list[float]]
    zone_half_width_rel: float = Field(ge=0)
    confluence_tol_rel: float = Field(ge=0)
    max_zones: int = Field(ge=1)
    time_ratios: list[float]
    invalidation_span: float = Field(gt=0)
    invalidation_iterations: int = Field(ge=5)
    path_ratio_index: int = Field(ge=0)

    def t(self, key: str) -> list[float]:
        """Return a list of projection ratios."""
        try:
            return list(self.targets[key])
        except KeyError as exc:  # pragma: no cover - config error path
            raise KeyError(f"Projektions-Verhältnisse '{key}' fehlen in der Konfiguration") from exc


class DegreeLabels(_Strict):
    motive: list[list[str]]
    corrective: list[list[str]]
    combination: list[list[str]]


class PlottingConfig(_Strict):
    scenario_colors: list[str]
    invalidation_color: str
    target_opacity: float = Field(ge=0, le=1)
    candle_up_color: str
    candle_down_color: str
    pivot_colors: list[str]
    label_font_sizes: list[int]
    top_degree: int = Field(ge=0)
    degree_labels: DegreeLabels
    height: int = Field(ge=200)
    volume_row_height: float = Field(gt=0, lt=1)


class Settings(_Strict):
    """Complete application configuration."""

    data: DataConfig
    indicators: IndicatorConfig
    pivots: PivotConfig
    search: SearchConfig
    rules: RulesConfig
    fib: FibConfig
    guidelines: GuidelineConfig
    scoring: ScoringConfig
    projection: ProjectionConfig
    plotting: PlottingConfig


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def default_config_dict() -> dict[str, Any]:
    """Load the packaged default configuration as a plain dict."""
    text = resources.files("elliott").joinpath(DEFAULT_CONFIG_RESOURCE).read_text(encoding="utf-8")
    data = yaml.safe_load(text)
    if not isinstance(data, dict):  # pragma: no cover - broken package
        raise ValueError("Standard-Konfiguration ist leer oder ungültig")
    return data


def load_settings(
    path: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Settings:
    """Load settings: defaults, then optional YAML file, then dict overrides."""
    data = default_config_dict()
    if path is not None:
        user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(user, dict):
            raise ValueError(f"Konfigurationsdatei {path} muss ein YAML-Mapping enthalten")
        data = deep_merge(data, user)
    if overrides:
        data = deep_merge(data, overrides)
    return Settings.model_validate(data)


def with_overrides(settings: Settings, overrides: dict[str, Any]) -> Settings:
    """Return a new Settings object with ``overrides`` deep-merged."""
    return Settings.model_validate(deep_merge(settings.model_dump(), overrides))
