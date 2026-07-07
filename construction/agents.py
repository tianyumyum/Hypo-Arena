"""Construction-pipeline agent factories: 4 roles in the Forge–Audit cycle."""

from __future__ import annotations

from agents import Agent, WebSearchTool

from basics import build_responses_agent
from basics import AuditResult, DomainConfig, SourceRecord

from .prompts import (
    context_audit_instructions,
    context_forge_instructions,
    hypothesis_audit_instructions,
    hypothesis_forge_instructions,
)
from .schema import ContextDraft, HypothesisDraft, HypothesisSetDraft


def context_forge_agent(domain: DomainConfig, *, profile_name: str = "gpt-5.4-high") -> Agent:
    """Forge for the Context stage; WebSearchTool enabled and forced per domain config.

    SDK's Agent.reset_tool_choice=True (default) means tool_choice='web_search' forces the
    first turn to call web_search, then auto-resets so the agent can compose the final
    structured ContextDraft.
    """
    enable_search = domain.context_search_enabled
    tools = [WebSearchTool()] if enable_search else []
    tool_choice = "web_search" if enable_search else None
    return build_responses_agent(
        instructions=context_forge_instructions(domain),
        name=f"context_forge@{domain.name}",
        output_type=ContextDraft,
        profile_name=profile_name,
        tool_choice=tool_choice,
        tools=tools,
    )


def context_audit_agent(domain: DomainConfig, *, profile_name: str = "gpt-5.4-high") -> Agent:
    """Auditor for the Context stage; no tools."""
    return build_responses_agent(
        instructions=context_audit_instructions(domain),
        name=f"context_audit@{domain.name}",
        output_type=AuditResult,
        profile_name=profile_name,
    )


def hypothesis_forge_agent(
    domain: DomainConfig,
    *,
    profile_name: str = "gpt-5.4-high",
    record: SourceRecord | None = None,
) -> Agent:
    """Forge for the Hypothesis stage; WebSearchTool enabled (and forced) when record carries analysis_urls."""
    output_type = HypothesisSetDraft if domain.multi_hypothesis else HypothesisDraft
    enable_search = bool(record and record.metadata.get("analysis_urls"))
    tools = [WebSearchTool()] if enable_search else []
    tool_choice = "web_search" if enable_search else None
    return build_responses_agent(
        instructions=hypothesis_forge_instructions(domain),
        name=f"hypothesis_forge@{domain.name}",
        output_type=output_type,
        profile_name=profile_name,
        tool_choice=tool_choice,
        tools=tools,
    )


def hypothesis_audit_agent(domain: DomainConfig, *, profile_name: str = "gpt-5.4-high") -> Agent:
    """Auditor for the Hypothesis stage; no tools."""
    return build_responses_agent(
        instructions=hypothesis_audit_instructions(domain),
        name=f"hypothesis_audit@{domain.name}",
        output_type=AuditResult,
        profile_name=profile_name,
    )
