"""Evaluation pipeline schemas: judge LLM outputs (wrapped into basics types by pipeline code)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

ArenaToken = Literal["A>>B", "A>B", "A=B", "B>A", "B>>A"]


class ArenaVerdict(BaseModel):
    """Arena-judge LLM output: a 5-level verdict + rationale."""

    rationale: str
    verdict: ArenaToken


class PairScore(BaseModel):
    """Per-Hypothesis rubric scores; field names mirror evaluation.rubric.PAIR_KEYS."""

    grounding: float
    insight: float
    justification: float


class SetScore(BaseModel):
    """Set-level rubric scores; field names mirror evaluation.rubric.SET_KEYS."""

    breadth: float
    distinctness: float | None                    # null when k<=1 (paper §3.1.2: N/A)
    utility: float


class ScoreVerdict(BaseModel):
    """Score-judge LLM output. All fields required (strict JSON); use null where N/A."""

    pair_scores: list[PairScore]                  # one entry per submitted Hypothesis
    rationale: str
    recall: str | None                            # null when no reference supplied
    set_scores: SetScore | None                   # null for non-multi-hypothesis domains
