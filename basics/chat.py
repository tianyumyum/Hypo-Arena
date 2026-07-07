"""Chat-Completions silo: pure factory for Agents on the 4-platform model pool."""

from __future__ import annotations

from agents import Agent, OpenAIChatCompletionsModel

from .models import GENERATION_REGISTRY, resolve_candidates
from .platform import FallbackModel


def build_chat_agent(
    *,
    instructions: str,
    name: str,
    output_type: type | None = None,
    profile_name: str,
) -> Agent:
    """Build an Agent on the Chat-Completions silo (pipeline-agnostic)."""
    if profile_name not in GENERATION_REGISTRY:
        raise KeyError(
            f"Unknown Chat profile: {profile_name!r}. "
            f"Known: {sorted(GENERATION_REGISTRY)}"
        )
    profile = GENERATION_REGISTRY[profile_name]
    model = FallbackModel(resolve_candidates(profile_name, profile, OpenAIChatCompletionsModel))
    return Agent(
        name=name,
        instructions=instructions,
        model=model,
        model_settings=profile.model_settings,
        output_type=output_type,
    )
