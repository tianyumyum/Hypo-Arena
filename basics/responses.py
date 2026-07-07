"""Responses-API silo: pure factory for Agents on the gpt-5 family."""

from __future__ import annotations

from dataclasses import replace

from agents import Agent, OpenAIResponsesModel
from agents.tool import Tool

from .models import CONSTRUCTION_REGISTRY, resolve_candidates
from .platform import FallbackModel


def build_responses_agent(
    *,
    instructions: str,
    name: str,
    output_type: type | None = None,
    profile_name: str = "gpt-5.4-high",
    tool_choice: str | None = None,
    tools: list[Tool] | None = None,
) -> Agent:
    """Build an Agent on the Responses-API silo (pipeline-agnostic; tools are caller's choice)."""
    if profile_name not in CONSTRUCTION_REGISTRY:
        raise KeyError(
            f"Unknown Responses profile: {profile_name!r}. "
            f"Known: {sorted(CONSTRUCTION_REGISTRY)}"
        )
    profile = CONSTRUCTION_REGISTRY[profile_name]
    model = FallbackModel(resolve_candidates(profile_name, profile, OpenAIResponsesModel))
    settings = profile.model_settings
    if tool_choice is not None:
        settings = replace(settings, tool_choice=tool_choice)
    return Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=settings,
        output_type=output_type,
        tools=tools or [],
    )
