"""Evaluation-pipeline agent factories: arena pairwise judge + score absolute judge."""

from __future__ import annotations

from agents import Agent

from basics import build_chat_agent
from basics import DomainConfig
from basics.models import GENERATION_REGISTRY

from .prompts import arena_judge_instructions, score_judge_instructions
from .schema import ArenaVerdict, ScoreVerdict


def _output_type_for(profile_name: str, structured: type):
    """Return the schema class for profiles that accept response_format=json_schema,
    else None (text mode; consumer must manually parse via basics.parsing).
    """
    profile = GENERATION_REGISTRY.get(profile_name)
    if profile is None or profile.supports_response_format:
        return structured
    return None


def arena_judge_agent(domain: DomainConfig, *, profile_name: str) -> Agent:
    """Pairwise arena judge: returns a 5-level verdict + rationale."""
    return build_chat_agent(
        instructions=arena_judge_instructions(domain),
        name=f"arena_judge@{domain.name}+{profile_name}",
        output_type=_output_type_for(profile_name, ArenaVerdict),
        profile_name=profile_name,
    )


def score_judge_agent(
    domain: DomainConfig,
    *,
    profile_name: str,
    with_recall: bool = False,
) -> Agent:
    """Absolute scoring judge; with_recall=True activates the reference-anchored recall block."""
    suffix = "+recall" if with_recall else ""
    return build_chat_agent(
        instructions=score_judge_instructions(domain, with_recall=with_recall),
        name=f"score_judge@{domain.name}+{profile_name}{suffix}",
        output_type=_output_type_for(profile_name, ScoreVerdict),
        profile_name=profile_name,
    )
