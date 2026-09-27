"""Combinations W-X-Y and W-X-Y-X-Z (sideways double/triple threes).

At most one triangle, and only as the last corrective (Y or Z).
"""

from __future__ import annotations

from elliott.guidelines import COMBINATION_GUIDES
from elliott.models import PatternType
from elliott.patterns.base import CORRECTIVE_ALL, FLATS, TRIANGLES, ZIGZAGS, C, PatternSpec
from elliott.rules import COMB_SIDEWAYS, COMB_TRIANGLES, COMB_X_MAX

_PART = ZIGZAGS | FLATS
_LAST = _PART | TRIANGLES

COMBINATION_WXY = PatternSpec(
    type=PatternType.COMBINATION_WXY, family="combination", label_style="combination",
    wave_classes=(C, C, C), sub_types=(_PART, CORRECTIVE_ALL, _LAST),
    rules=(COMB_X_MAX, COMB_SIDEWAYS, COMB_TRIANGLES), guidelines=COMBINATION_GUIDES,
)
COMBINATION_WXYXZ = PatternSpec(
    type=PatternType.COMBINATION_WXYXZ, family="combination", label_style="combination",
    wave_classes=(C, C, C, C, C), sub_types=(_PART, CORRECTIVE_ALL, _PART, CORRECTIVE_ALL, _LAST),
    rules=(COMB_X_MAX, COMB_SIDEWAYS, COMB_TRIANGLES), guidelines=COMBINATION_GUIDES,
)

SPECS = (COMBINATION_WXY, COMBINATION_WXYXZ)
