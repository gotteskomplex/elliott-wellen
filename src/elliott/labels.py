"""Wave label conventions per degree (①②③ / (1)(2)(3) / 1 2 3 / i ii iii ...)."""

from __future__ import annotations

from elliott.patterns import PatternSpec
from elliott.settings import PlottingConfig


def display_label(spec: PatternSpec, wave: int, degree: int, cfg: PlottingConfig) -> str:
    """Display label of wave ``wave`` (1-based) of ``spec`` at relative ``degree``."""
    styles = getattr(cfg.degree_labels, spec.label_style)
    row = styles[min(degree, len(styles) - 1)]
    return row[wave - 1] if wave - 1 < len(row) else spec.labels[wave - 1]
