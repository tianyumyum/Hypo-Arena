"""Forge intermediate schemas, merged into basics.BenchmarkCase fields after audit passes."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ContextDraft(BaseModel):
    """Forge output for the Context stage."""

    context: str


class HypothesisDraft(BaseModel):
    """Forge output for single-hypothesis (research) Hypothesis stage."""

    evidence: str
    hypothesis: str


class CategorizedHypothesis(BaseModel):
    """Multi-hypothesis Forge item: every entry carries a non-empty analytical category."""

    category: str = Field(min_length=1)
    evidence: str
    hypothesis: str


class HypothesisSetDraft(BaseModel):
    """Forge output for multi-hypothesis (real-world) Hypothesis stage."""

    hypotheses: list[CategorizedHypothesis]
