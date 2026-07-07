"""Forge–Audit loop that turns one SourceRecord into one BenchmarkCase."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agents import Runner
from agents.result import RunResult
from openai.types.responses import ResponseFunctionWebSearch

from basics import (
    AuditResult,
    BenchmarkCase,
    CaseQuality,
    ConstructionProvenance,
    DomainConfig,
    HypothesisItem,
    SourceRecord,
    TokenUsage,
    WebSearchTrace,
)

from .agents import (
    context_audit_agent,
    context_forge_agent,
    hypothesis_audit_agent,
    hypothesis_forge_agent,
)
from .prompts import (
    context_audit_prompt,
    context_forge_initial_prompt,
    context_revision_prompt,
    hypothesis_audit_prompt,
    hypothesis_forge_initial_prompt,
    hypothesis_revision_prompt,
)
from .schema import ContextDraft, HypothesisDraft, HypothesisSetDraft
from .source_input import assemble_forge_input

logger = logging.getLogger("hypo.construction")


DEFAULT_MAX_ROUNDS = 4


@dataclass
class ForgeAuditStageResult:
    """Output of one Forge–Audit sub-loop (Context or Hypothesis)."""

    audit: AuditResult
    final_output: Any
    rounds: int
    web_searches: list[WebSearchTrace]


def _extract_usage(run: RunResult) -> TokenUsage:
    """Pull token counters out of a Runner.run result."""
    return TokenUsage.from_sdk_usage(run.context_wrapper.usage)


def _extract_web_searches(run: RunResult) -> list[WebSearchTrace]:
    """Pull every web_search tool call from a Runner.run result's new_items."""
    traces: list[WebSearchTrace] = []
    for item in run.new_items:
        raw = getattr(item, "raw_item", None)
        if not isinstance(raw, ResponseFunctionWebSearch):
            continue
        action = raw.action
        kind = action.type
        query: str | None = None
        url: str | None = None
        if kind == "search":
            query = action.query
        elif kind == "open_page":
            url = action.url
        elif kind == "find_in_page":
            query = action.pattern
            url = action.url
        traces.append(WebSearchTrace(action=kind, query=query, url=url))
    return traces


def _forge_history(forge_run: RunResult, revision_prompt: str) -> list[Any]:
    """Carry the Forge's full output history forward and append the next revision prompt."""
    history = list(forge_run.to_input_list())
    history.append({"role": "user", "content": revision_prompt})
    return history


async def _run_forge_audit(
    *,
    audit,
    audit_prompt_fn,
    first_forge_input,
    forge,
    max_rounds: int,
    revision_prompt_fn,
) -> tuple[ForgeAuditStageResult, TokenUsage]:
    """Generic Forge–Audit loop: alternates Forge draft and Audit review until pass or cap."""
    tokens = TokenUsage()
    web_searches: list[WebSearchTrace] = []
    forge_input: Any = first_forge_input
    forge_run: RunResult | None = None
    audit_result: AuditResult | None = None
    final_output: Any = None

    for round_index in range(1, max_rounds + 1):
        forge_run = await Runner.run(forge, forge_input)
        tokens = tokens + _extract_usage(forge_run)
        web_searches.extend(_extract_web_searches(forge_run))
        final_output = forge_run.final_output
        audit_run = await Runner.run(audit, audit_prompt_fn(final_output))
        tokens = tokens + _extract_usage(audit_run)
        audit_result = audit_run.final_output
        logger.info(
            "forge_audit round=%d passed=%s issues=%d",
            round_index, audit_result.passed, len(audit_result.issues),
        )
        if audit_result.passed:
            break
        forge_input = _forge_history(
            forge_run,
            revision_prompt_fn(audit_result, round_number=round_index + 1),
        )

    assert audit_result is not None
    return (
        ForgeAuditStageResult(
            audit=audit_result,
            final_output=final_output,
            rounds=round_index,
            web_searches=web_searches,
        ),
        tokens,
    )


async def run_context_stage(
    *,
    domain: DomainConfig,
    forge_profile: str,
    max_rounds: int,
    record: SourceRecord,
) -> tuple[ForgeAuditStageResult, TokenUsage]:
    """Forge then Audit the Context until it passes or max_rounds is reached."""
    forge = context_forge_agent(domain, profile_name=forge_profile)
    audit = context_audit_agent(domain, profile_name=forge_profile)

    def _audit_prompt(ctx_draft: ContextDraft) -> str:
        return context_audit_prompt(ctx_draft.context, record)

    return await _run_forge_audit(
        audit=audit,
        audit_prompt_fn=_audit_prompt,
        first_forge_input=assemble_forge_input(context_forge_initial_prompt(record), record),
        forge=forge,
        max_rounds=max_rounds,
        revision_prompt_fn=context_revision_prompt,
    )


async def run_hypothesis_stage(
    *,
    context: str,
    domain: DomainConfig,
    forge_profile: str,
    max_rounds: int,
    record: SourceRecord,
) -> tuple[ForgeAuditStageResult, TokenUsage]:
    """Forge then Audit the Hypothesis until it passes or max_rounds is reached."""
    forge = hypothesis_forge_agent(domain, profile_name=forge_profile, record=record)
    audit = hypothesis_audit_agent(domain, profile_name=forge_profile)

    def _audit_prompt(draft: HypothesisDraft | HypothesisSetDraft) -> str:
        return hypothesis_audit_prompt(context, draft.model_dump(), record)

    return await _run_forge_audit(
        audit=audit,
        audit_prompt_fn=_audit_prompt,
        first_forge_input=assemble_forge_input(
            hypothesis_forge_initial_prompt(context, record), record,
        ),
        forge=forge,
        max_rounds=max_rounds,
        revision_prompt_fn=hypothesis_revision_prompt,
    )


def _draft_to_hypotheses(draft: HypothesisDraft | HypothesisSetDraft) -> list[HypothesisItem]:
    """Normalize either draft shape into the unified HypothesisItem list."""
    if isinstance(draft, HypothesisSetDraft):
        return [
            HypothesisItem(category=h.category, evidence=h.evidence, hypothesis=h.hypothesis)
            for h in draft.hypotheses
        ]
    return [HypothesisItem(hypothesis=draft.hypothesis, evidence=draft.evidence)]


async def construct_case(
    *,
    domain: DomainConfig,
    forge_profile: str = "gpt-5.4-high",
    max_rounds: int = DEFAULT_MAX_ROUNDS,
    record: SourceRecord,
) -> BenchmarkCase:
    """End-to-end construction of one BenchmarkCase from one SourceRecord."""
    logger.info("construct.start id=%s profile=%s", record.id, forge_profile)

    context_stage, context_tokens = await run_context_stage(
        domain=domain, forge_profile=forge_profile, max_rounds=max_rounds, record=record,
    )
    context_text: str = context_stage.final_output.context

    hypothesis_stage, hypothesis_tokens = await run_hypothesis_stage(
        context=context_text,
        domain=domain,
        forge_profile=forge_profile,
        max_rounds=max_rounds,
        record=record,
    )
    hypotheses = _draft_to_hypotheses(hypothesis_stage.final_output)

    quality = CaseQuality(
        context_audit=context_stage.audit,
        hypothesis_audit=hypothesis_stage.audit,
    )
    provenance = ConstructionProvenance(
        context_rounds=context_stage.rounds,
        hypothesis_rounds=hypothesis_stage.rounds,
        profile=forge_profile,
        tokens=context_tokens + hypothesis_tokens,
        web_searches=context_stage.web_searches + hypothesis_stage.web_searches,
    )

    case = BenchmarkCase(
        context=context_text,
        domain=domain.name,
        hypotheses=hypotheses,
        id=record.id,
        metadata=dict(record.metadata),
        provenance=provenance,
        quality=quality,
    )
    logger.info(
        "construct.end id=%s passed=%s rounds=(%d,%d)",
        record.id, quality.passed, context_stage.rounds, hypothesis_stage.rounds,
    )
    return case


def encode_case(case: BenchmarkCase) -> str:
    """Serialize a BenchmarkCase to a single JSON line."""
    return json.dumps(case.model_dump(mode="json"), ensure_ascii=False)
