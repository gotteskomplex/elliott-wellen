"""Domain models (Pydantic) shared by the analysis core, reports and UI.

The search itself works on lightweight numpy structures for speed (see
:mod:`elliott.measure`); results are converted into these models at the end.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class PivotKind(str, Enum):
    """Type of a turning point."""

    HIGH = "high"
    LOW = "low"

    @property
    def sign(self) -> int:
        """+1 for highs, -1 for lows."""
        return 1 if self is PivotKind.HIGH else -1

    @classmethod
    def from_sign(cls, sign: int) -> "PivotKind":
        return cls.HIGH if sign > 0 else cls.LOW


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.UP else -1

    @classmethod
    def from_sign(cls, sign: int) -> "Direction":
        return cls.UP if sign > 0 else cls.DOWN

    @property
    def german(self) -> str:
        return "aufwärts" if self is Direction.UP else "abwärts"


class PatternType(str, Enum):
    """All supported Elliott patterns."""

    IMPULSE = "impulse"
    LEADING_DIAGONAL = "leading_diagonal"
    ENDING_DIAGONAL = "ending_diagonal"
    ZIGZAG = "zigzag"
    DOUBLE_ZIGZAG = "double_zigzag"
    TRIPLE_ZIGZAG = "triple_zigzag"
    FLAT_REGULAR = "flat_regular"
    FLAT_EXPANDED = "flat_expanded"
    FLAT_RUNNING = "flat_running"
    TRIANGLE_CONTRACTING = "triangle_contracting"
    TRIANGLE_BARRIER = "triangle_barrier"
    TRIANGLE_EXPANDING = "triangle_expanding"
    COMBINATION_WXY = "combination_wxy"
    COMBINATION_WXYXZ = "combination_wxyxz"

    @property
    def german(self) -> str:
        return PATTERN_NAMES_DE[self]


PATTERN_NAMES_DE: dict[PatternType, str] = {
    PatternType.IMPULSE: "Impuls",
    PatternType.LEADING_DIAGONAL: "Leading Diagonal",
    PatternType.ENDING_DIAGONAL: "Ending Diagonal",
    PatternType.ZIGZAG: "Zigzag",
    PatternType.DOUBLE_ZIGZAG: "Double Zigzag",
    PatternType.TRIPLE_ZIGZAG: "Triple Zigzag",
    PatternType.FLAT_REGULAR: "Flat (regular)",
    PatternType.FLAT_EXPANDED: "Flat (expanded)",
    PatternType.FLAT_RUNNING: "Flat (running)",
    PatternType.TRIANGLE_CONTRACTING: "Dreieck (contracting)",
    PatternType.TRIANGLE_BARRIER: "Dreieck (barrier)",
    PatternType.TRIANGLE_EXPANDING: "Dreieck (expanding)",
    PatternType.COMBINATION_WXY: "Kombination W-X-Y",
    PatternType.COMBINATION_WXYXZ: "Kombination W-X-Y-X-Z",
}


class WaveClass(str, Enum):
    """Motive (trend) or corrective (counter-trend) wave."""

    MOTIVE = "motive"
    CORRECTIVE = "corrective"


class SubwaveStatus(str, Enum):
    VALIDATED = "validated"
    NOT_VALIDATED = "not_validated"
    FAILED = "failed"

    @property
    def german(self) -> str:
        return {
            SubwaveStatus.VALIDATED: "validiert",
            SubwaveStatus.NOT_VALIDATED: "nicht validiert",
            SubwaveStatus.FAILED: "nicht bestätigt",
        }[self]


class _Model(BaseModel):
    model_config = ConfigDict(frozen=False, use_enum_values=False)


class Pivot(_Model):
    """A detected turning point."""

    idx: int = Field(description="Bar index in the price data")
    ts: datetime
    price: float
    kind: PivotKind
    level: int = Field(description="Degree level, 0 = coarsest")
    provisional: bool = Field(default=False, description="Unconfirmed last pivot at the right edge")


class RuleCheck(_Model):
    """Result of one hard rule for a count."""

    name: str
    passed: bool
    reason: str
    deferred: bool = False


class GuidelineScore(_Model):
    """Contribution of one soft guideline to the score."""

    name: str
    title: str
    weight: float
    value: float | None = Field(description="0..1, None = not applicable")
    points: float
    reason: str


class Penalty(_Model):
    name: str
    points: float
    reason: str


class ScoreBreakdown(_Model):
    total: float
    guideline_score: float
    guidelines: list[GuidelineScore]
    penalties: list[Penalty]
    complexity: float


class Wave(_Model):
    """One wave of a count."""

    label: str
    display_label: str
    start: Pivot
    end: Pivot
    direction: Direction
    wave_class: WaveClass
    length_price: float
    length_measure: float = Field(description="Length in the measurement scale (linear or log)")
    duration_bars: int
    ratios: dict[str, float] = Field(default_factory=dict)
    open: bool = False
    subwave_status: SubwaveStatus = SubwaveStatus.NOT_VALIDATED
    subwave_note: str = ""
    subcount: "WaveCount | None" = None


class WaveCount(_Model):
    """A (possibly incomplete) count of one pattern."""

    pattern: PatternType
    direction: Direction
    degree: int
    waves: list[Wave]
    expected_waves: int
    complete: bool

    @property
    def labels(self) -> list[str]:
        return [w.label for w in self.waves]


class TargetZone(_Model):
    low: float
    high: float
    center: float
    sources: list[str]
    confluence: int
    highlighted: bool


class Invalidation(_Model):
    price: float
    side: str = Field(description="'below' or 'above'")
    rule: str
    reason: str
    hard: bool = True


class TimeWindow(_Model):
    earliest: datetime
    latest: datetime
    note: str


class Projection(_Model):
    position: str
    current_wave: str
    current_wave_running: bool
    next_move: str
    next_direction: Direction | None
    expected_pattern: str
    targets: list[TargetZone]
    invalidation: Invalidation
    invalidations: list[Invalidation]
    alternative: str = ""
    alternative_scenario_id: str | None = None
    time_window: TimeWindow | None = None
    path: list[tuple[datetime, float]] = Field(default_factory=list)
    sub_position: str = ""


class Scenario(_Model):
    id: str
    rank: int
    count: WaveCount
    score: ScoreBreakdown
    probability: float | None = Field(default=None, description="Heuristic softmax weight")
    projection: Projection | None = None
    rule_checks: list[RuleCheck]
    notes: list[str] = Field(default_factory=list)
    start_pivot_level: int


class LevelInfo(_Model):
    level: int
    name: str
    threshold: str
    pivot_count: int


class AnalysisMeta(_Model):
    source: str
    start: datetime
    end: datetime
    bars: int
    log_scale: bool
    last_close: float
    has_volume: bool
    levels: list[LevelInfo]
    search_level: int
    runtime_s: float
    states_explored: int
    budget_exhausted: bool


class AnalysisResult(_Model):
    meta: AnalysisMeta
    pivots: dict[int, list[Pivot]]
    scenarios: list[Scenario]
    warnings: list[str]
    disclaimer: str


Wave.model_rebuild()
