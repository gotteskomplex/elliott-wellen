"""Flat (3-3-5): regular, expanded and running."""

from __future__ import annotations

from elliott.guidelines import FLAT_GUIDES
from elliott.models import PatternType
from elliott.patterns.base import CORRECTIVE_NO_TRIANGLE, LAST_WAVE_MOTIVE, C, M, PatternSpec
from elliott.rules import FLAT_B_MAX, FLAT_B_MIN, FLAT_EXP_C, FLAT_IRR_B, FLAT_REG_B, FLAT_REG_C, FLAT_RUN_C

_SUBS = (CORRECTIVE_NO_TRIANGLE, CORRECTIVE_NO_TRIANGLE, LAST_WAVE_MOTIVE)
_CLASSES = (C, C, M)

FLAT_REGULAR = PatternSpec(
    type=PatternType.FLAT_REGULAR,
    family="flat",
    label_style="corrective",
    wave_classes=_CLASSES,
    sub_types=_SUBS,
    rules=(FLAT_B_MIN, FLAT_B_MAX, FLAT_REG_B, FLAT_REG_C),
    guidelines=FLAT_GUIDES,
)

FLAT_EXPANDED = PatternSpec(
    type=PatternType.FLAT_EXPANDED,
    family="flat",
    label_style="corrective",
    wave_classes=_CLASSES,
    sub_types=_SUBS,
    rules=(FLAT_B_MIN, FLAT_B_MAX, FLAT_IRR_B, FLAT_EXP_C),
    guidelines=FLAT_GUIDES,
)

FLAT_RUNNING = PatternSpec(
    type=PatternType.FLAT_RUNNING,
    family="flat",
    label_style="corrective",
    wave_classes=_CLASSES,
    sub_types=_SUBS,
    rules=(FLAT_B_MIN, FLAT_B_MAX, FLAT_IRR_B, FLAT_RUN_C),
    guidelines=FLAT_GUIDES,
    penalty="running_flat",
)

SPECS = (FLAT_REGULAR, FLAT_EXPANDED, FLAT_RUNNING)
