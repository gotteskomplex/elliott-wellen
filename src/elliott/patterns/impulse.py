"""Impulse (5-3-5-3-5)."""

from __future__ import annotations

from elliott.guidelines import IMPULSE_GUIDES
from elliott.models import PatternType
from elliott.patterns.base import (
    CORRECTIVE_ALL, CORRECTIVE_NO_TRIANGLE, FIRST_WAVE_MOTIVE, IMPULSE_ONLY, LAST_WAVE_MOTIVE, C, M, PatternSpec,
)
from elliott.rules import W2_MAX, W3_BEYOND_W1, W3_NOT_SHORTEST, W4_NO_OVERLAP

IMPULSE = PatternSpec(
    type=PatternType.IMPULSE,
    family="impulse",
    label_style="motive",
    wave_classes=(M, C, M, C, M),
    # Wave 2 is never a triangle; wave 1 may be a leading, wave 5 an ending diagonal.
    sub_types=(FIRST_WAVE_MOTIVE, CORRECTIVE_NO_TRIANGLE, IMPULSE_ONLY, CORRECTIVE_ALL, LAST_WAVE_MOTIVE),
    rules=(W2_MAX, W3_BEYOND_W1, W4_NO_OVERLAP, W3_NOT_SHORTEST),
    guidelines=IMPULSE_GUIDES,
)

SPECS = (IMPULSE,)
