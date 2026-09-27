"""Leading diagonal (5-3-5-3-5 or 3-3-3-3-3) and ending diagonal (3-3-3-3-3).

Both share the same price rules; they differ in their allowed position in the
larger pattern (leading: wave 1/A, ending: wave 5/C – enforced via the
``sub_types`` of the parent patterns) and in their internal structure.
"""

from __future__ import annotations

from elliott.guidelines import DIAGONAL_GUIDES
from elliott.models import PatternType
from elliott.patterns.base import CORRECTIVE_NO_TRIANGLE, IMPULSE_ONLY, ZIGZAGS, C, M, PatternSpec
from elliott.rules import (
    DIAG_LINES, DIAG_SHAPE, DIAG_W4_MAX, DIAG_W4_OVERLAP, W2_MAX, W3_BEYOND_W1, W3_NOT_SHORTEST,
)

_RULES = (W2_MAX, W3_BEYOND_W1, DIAG_W4_MAX, DIAG_W4_OVERLAP, W3_NOT_SHORTEST, DIAG_SHAPE, DIAG_LINES)
_MOTIVE_OR_THREE = IMPULSE_ONLY | ZIGZAGS

LEADING_DIAGONAL = PatternSpec(
    type=PatternType.LEADING_DIAGONAL,
    family="diagonal",
    label_style="motive",
    wave_classes=(M, C, M, C, M),
    sub_types=(_MOTIVE_OR_THREE, CORRECTIVE_NO_TRIANGLE, _MOTIVE_OR_THREE, CORRECTIVE_NO_TRIANGLE, _MOTIVE_OR_THREE),
    rules=_RULES,
    guidelines=DIAGONAL_GUIDES,
)

ENDING_DIAGONAL = PatternSpec(
    type=PatternType.ENDING_DIAGONAL,
    family="diagonal",
    label_style="motive",
    wave_classes=(M, C, M, C, M),
    sub_types=(ZIGZAGS, ZIGZAGS, ZIGZAGS, ZIGZAGS, ZIGZAGS),
    rules=_RULES,
    guidelines=DIAGONAL_GUIDES,
)

SPECS = (LEADING_DIAGONAL, ENDING_DIAGONAL)
