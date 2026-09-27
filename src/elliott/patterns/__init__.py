"""Pattern registry."""

from __future__ import annotations

from elliott.models import PatternType
from elliott.patterns import combination, diagonal, flat, impulse, triangle, zigzag
from elliott.patterns.base import PatternSpec

SPECS: dict[PatternType, PatternSpec] = {
    spec.type: spec
    for module in (impulse, diagonal, zigzag, flat, triangle, combination)
    for spec in module.SPECS
}


def get_spec(pattern: PatternType) -> PatternSpec:
    """Return the specification of a pattern type."""
    return SPECS[pattern]


def all_specs() -> list[PatternSpec]:
    return list(SPECS.values())


__all__ = ["SPECS", "PatternSpec", "all_specs", "get_spec"]
