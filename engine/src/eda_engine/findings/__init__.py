"""Findings engine: statistically tested, false-discovery-controlled findings."""

from eda_engine.findings.engine import (
    Finding,
    FindingsConfig,
    FindingsReport,
    benjamini_hochberg,
    discover_findings,
    quality_notes,
)

__all__ = [
    "Finding",
    "FindingsConfig",
    "FindingsReport",
    "benjamini_hochberg",
    "discover_findings",
    "quality_notes",
]
