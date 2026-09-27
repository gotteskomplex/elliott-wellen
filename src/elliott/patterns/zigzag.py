"""Zigzag (5-3-5) plus double (W-X-Y) and triple (W-X-Y-X-Z) zigzag."""

from __future__ import annotations

from elliott.guidelines import DOUBLE_ZIGZAG_GUIDES, ZIGZAG_GUIDES
from elliott.models import PatternType
from elliott.patterns.base import CORRECTIVE_ALL, FIRST_WAVE_MOTIVE, LAST_WAVE_MOTIVE, C, M, PatternSpec
from elliott.rules import TZ_X2_MAX, TZ_Z_BEYOND, ZZ_B_MAX, ZZ_C_BEYOND

_ZZ = frozenset({PatternType.ZIGZAG})

ZIGZAG = PatternSpec(
    type=PatternType.ZIGZAG,
    family="zigzag",
    label_style="corrective",
    wave_classes=(M, C, M),
    sub_types=(FIRST_WAVE_MOTIVE, CORRECTIVE_ALL, LAST_WAVE_MOTIVE),
    rules=(ZZ_B_MAX, ZZ_C_BEYOND),
    guidelines=ZIGZAG_GUIDES,
)

# Double zigzag: price rules of a zigzag at the higher degree, but W and Y are
# themselves zigzags (3-wave), connected by any corrective X.
DOUBLE_ZIGZAG = PatternSpec(
    type=PatternType.DOUBLE_ZIGZAG,
    family="zigzag",
    label_style="combination",
    wave_classes=(C, C, C),
    sub_types=(_ZZ, CORRECTIVE_ALL, _ZZ),
    rules=(ZZ_B_MAX, ZZ_C_BEYOND),
    guidelines=DOUBLE_ZIGZAG_GUIDES,
)

TRIPLE_ZIGZAG = PatternSpec(
    type=PatternType.TRIPLE_ZIGZAG,
    family="zigzag",
    label_style="combination",
    wave_classes=(C, C, C, C, C),
    sub_types=(_ZZ, CORRECTIVE_ALL, _ZZ, CORRECTIVE_ALL, _ZZ),
    rules=(ZZ_B_MAX, ZZ_C_BEYOND, TZ_X2_MAX, TZ_Z_BEYOND),
    guidelines=DOUBLE_ZIGZAG_GUIDES,
)

SPECS = (ZIGZAG, DOUBLE_ZIGZAG, TRIPLE_ZIGZAG)
