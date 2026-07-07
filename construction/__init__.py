"""Construction stage: Forge–Audit loop turning SourceRecord → BenchmarkCase."""

from .agents import (
    context_audit_agent,
    context_forge_agent,
    hypothesis_audit_agent,
    hypothesis_forge_agent,
)
from .runtime import (
    DEFAULT_MAX_ROUNDS,
    construct_case,
    encode_case,
    run_context_stage,
    run_hypothesis_stage,
)
from .schema import CategorizedHypothesis, ContextDraft, HypothesisDraft, HypothesisSetDraft

__all__ = [
    "CategorizedHypothesis",
    "ContextDraft",
    "DEFAULT_MAX_ROUNDS",
    "HypothesisDraft",
    "HypothesisSetDraft",
    "construct_case",
    "context_audit_agent",
    "context_forge_agent",
    "encode_case",
    "hypothesis_audit_agent",
    "hypothesis_forge_agent",
    "run_context_stage",
    "run_hypothesis_stage",
]
