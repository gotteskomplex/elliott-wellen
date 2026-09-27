"""Pattern specification shared by all pattern modules."""

from __future__ import annotations

from dataclasses import dataclass

from elliott.guidelines import GENERAL_GUIDES, Guideline
from elliott.models import PATTERN_LABELS, PatternType, WaveClass
from elliott.rules import ALTERNATION, SUBWAVE_TYPES, Rule

M, C = WaveClass.MOTIVE, WaveClass.CORRECTIVE

# Groups of pattern types used to describe allowed subwave structures.
ZIGZAGS = frozenset({PatternType.ZIGZAG, PatternType.DOUBLE_ZIGZAG, PatternType.TRIPLE_ZIGZAG})
FLATS = frozenset({PatternType.FLAT_REGULAR, PatternType.FLAT_EXPANDED, PatternType.FLAT_RUNNING})
TRIANGLES = frozenset(
    {PatternType.TRIANGLE_CONTRACTING, PatternType.TRIANGLE_BARRIER, PatternType.TRIANGLE_EXPANDING}
)
COMBINATIONS = frozenset({PatternType.COMBINATION_WXY, PatternType.COMBINATION_WXYXZ})
CORRECTIVE_NO_TRIANGLE = ZIGZAGS | FLATS | COMBINATIONS
CORRECTIVE_ALL = CORRECTIVE_NO_TRIANGLE | TRIANGLES
FIRST_WAVE_MOTIVE = frozenset({PatternType.IMPULSE, PatternType.LEADING_DIAGONAL})
LAST_WAVE_MOTIVE = frozenset({PatternType.IMPULSE, PatternType.ENDING_DIAGONAL})
IMPULSE_ONLY = frozenset({PatternType.IMPULSE})


@dataclass(frozen=True)
class PatternSpec:
    """Static description of one Elliott pattern.

    Attributes:
        type: the pattern type.
        family: grouping for de-duplication of alternatives on identical pivots.
        label_style: which label set (motive/corrective/combination) to display.
        wave_classes: motive/corrective role of each wave within the pattern.
        sub_types: allowed pattern types of each subwave (context rules).
        rules: hard rules (the common ones are added automatically).
        guidelines: pattern specific soft guidelines (general ones are added).
        penalty: optional key into ``guidelines.penalties`` for rare patterns.
    """

    type: PatternType
    family: str
    label_style: str
    wave_classes: tuple[WaveClass, ...]
    sub_types: tuple[frozenset[PatternType], ...]
    rules: tuple[Rule, ...]
    guidelines: tuple[Guideline, ...]
    penalty: str | None = None

    def __post_init__(self) -> None:
        n = len(PATTERN_LABELS[self.type])
        if len(self.wave_classes) != n or len(self.sub_types) != n:
            raise ValueError(f"Inkonsistente Spezifikation für {self.type}")

    @property
    def n(self) -> int:
        return len(self.wave_classes)

    @property
    def labels(self) -> tuple[str, ...]:
        return PATTERN_LABELS[self.type]

    @property
    def is_motive(self) -> bool:
        return self.family in ("impulse", "diagonal")

    @property
    def all_rules(self) -> tuple[Rule, ...]:
        """Common structural rules first (cheap), then pattern rules, then context."""
        return (ALTERNATION, *self.rules, SUBWAVE_TYPES)

    @property
    def all_guidelines(self) -> tuple[Guideline, ...]:
        return (*self.guidelines, *GENERAL_GUIDES)

    def relaxed_start(self, wave: int) -> bool:
        """Corrective waves may have an irregular start (e.g. expanded-flat B)."""
        return self.wave_classes[wave - 1] is WaveClass.CORRECTIVE
