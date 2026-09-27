"""Triangles (3-3-3-3-3): contracting, barrier, expanding.

Triangles only occur as wave 4, B, X or as the last wave of a combination –
never as wave 2. This is enforced through the ``sub_types`` of the parent
patterns.
"""

from __future__ import annotations

from elliott.guidelines import TRIANGLE_GUIDES
from elliott.models import PatternType
from elliott.patterns.base import CORRECTIVE_ALL, CORRECTIVE_NO_TRIANGLE, C, PatternSpec
from elliott.rules import TRI_BARRIER, TRI_CONTRACTING, TRI_EXPANDING

_SUBS = (CORRECTIVE_NO_TRIANGLE,) * 4 + (CORRECTIVE_ALL,)
_CLASSES = (C, C, C, C, C)

TRIANGLE_CONTRACTING = PatternSpec(
    type=PatternType.TRIANGLE_CONTRACTING, family="triangle", label_style="corrective",
    wave_classes=_CLASSES, sub_types=_SUBS, rules=(TRI_CONTRACTING,), guidelines=TRIANGLE_GUIDES,
)
TRIANGLE_BARRIER = PatternSpec(
    type=PatternType.TRIANGLE_BARRIER, family="triangle", label_style="corrective",
    wave_classes=_CLASSES, sub_types=_SUBS, rules=(TRI_BARRIER,), guidelines=TRIANGLE_GUIDES,
)
TRIANGLE_EXPANDING = PatternSpec(
    type=PatternType.TRIANGLE_EXPANDING, family="triangle", label_style="corrective",
    wave_classes=_CLASSES, sub_types=_SUBS, rules=(TRI_EXPANDING,), guidelines=TRIANGLE_GUIDES,
    penalty="expanding_triangle",
)

SPECS = (TRIANGLE_CONTRACTING, TRIANGLE_BARRIER, TRIANGLE_EXPANDING)
