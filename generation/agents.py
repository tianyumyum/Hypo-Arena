"""Generation-pipeline agent factories: Baseline writer, Agent-Mode selector + writers."""

from __future__ import annotations

from agents import Agent

from basics import build_chat_agent
from basics import DomainConfig
from basics.models import GENERATION_REGISTRY

from .prompts import (
    agent_intermediate_instructions,
    agent_writer_instructions,
    base_writer_instructions,
    skill_selector_instructions,
)
from .schema import FlatAgentOutputSchema, HypothesisCandidate, HypothesisCandidateSet, SkillSelection


def _writer_output_type(domain: DomainConfig) -> type:
    """Pick the structured-output type to match the domain's hypothesis cardinality."""
    return HypothesisCandidateSet if domain.multi_hypothesis else HypothesisCandidate


def _output_type_for(profile_name: str, structured: type):
    """Return the schema class for profiles that accept response_format=json_schema,
    else None (text mode; consumer must manually parse via basics.parsing).

    For gemini profiles, wrap in FlatAgentOutputSchema to inline `$defs`/`$ref`
    (Gemini's API rejects schemas containing reference pointers).
    """
    profile = GENERATION_REGISTRY.get(profile_name)
    if profile is None or profile.supports_response_format:
        if "gemini" in profile_name.lower():
            return FlatAgentOutputSchema(structured)
        return structured
    return None


def base_writer_agent(domain: DomainConfig, *, profile_name: str) -> Agent:
    """Single-pass Baseline-Mode writer."""
    return build_chat_agent(
        instructions=base_writer_instructions(domain),
        name=f"base_writer@{domain.name}+{profile_name}",
        output_type=_output_type_for(profile_name, _writer_output_type(domain)),
        profile_name=profile_name,
    )


def skill_selector_agent(domain: DomainConfig, *, profile_name: str) -> Agent:
    """Picks 0–3 skills (or 'none') for Agent Mode; output is parsed downstream."""
    return build_chat_agent(
        instructions=skill_selector_instructions(domain),
        name=f"skill_selector@{domain.name}+{profile_name}",
        output_type=_output_type_for(profile_name, SkillSelection),
        profile_name=profile_name,
    )


def agent_intermediate_writer(
    domain: DomainConfig,
    *,
    profile_name: str,
    skill_framework: str,
    skill_name: str,
    stage_index: int,
    stage_total: int,
) -> Agent:
    """Intermediate Agent-Mode pipeline stage; emits prose, no structured output."""
    return build_chat_agent(
        instructions=agent_intermediate_instructions(
            domain,
            skill_framework=skill_framework,
            stage_index=stage_index,
            stage_total=stage_total,
        ),
        name=f"agent_stage_{stage_index}[{skill_name}]@{domain.name}+{profile_name}",
        profile_name=profile_name,
    )


def agent_final_writer(
    domain: DomainConfig,
    *,
    profile_name: str,
    skill_framework: str,
    skill_name: str,
) -> Agent:
    """Final Agent-Mode pipeline stage; emits structured Hypothesis output."""
    return build_chat_agent(
        instructions=agent_writer_instructions(domain, skill_framework=skill_framework),
        name=f"agent_final[{skill_name}]@{domain.name}+{profile_name}",
        output_type=_output_type_for(profile_name, _writer_output_type(domain)),
        profile_name=profile_name,
    )
