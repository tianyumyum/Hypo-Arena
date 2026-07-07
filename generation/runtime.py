"""Generation pipeline: Baseline (single pass) and Agent (skill-driven) modes."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

from agents import Runner

from basics import (
    BenchmarkCase,
    DomainConfig,
    GenerationProvenance,
    HypothesisItem,
    Submission,
    TokenUsage,
)
from basics.parsing import coerce_to_model

from .agents import (
    agent_final_writer,
    agent_intermediate_writer,
    base_writer_agent,
    skill_selector_agent,
)
from .prompts import (
    agent_final_prompt,
    agent_intermediate_prompt,
    base_writer_prompt,
    selector_prompt,
)
from .schema import (
    CategorizedHypothesisCandidate,
    HypothesisCandidate,
    HypothesisCandidateSet,
    SkillSelection,
)
from .skills import SKILL_NAMES, SKILLS, Skill

logger = logging.getLogger("hypo.generation")


MAX_SKILLS_PER_CASE = 3


@dataclass
class SkillPipelineTrace:
    """Diagnostic trace of an Agent-Mode run for one case."""

    final_skill: str | None
    skills_selected: list[str]
    skills_used: list[str]


def _extract_usage(run) -> TokenUsage:
    """Pull token counters out of a Runner.run result."""
    return TokenUsage.from_sdk_usage(run.context_wrapper.usage)


def _candidate_to_items(
    output: HypothesisCandidate | HypothesisCandidateSet,
) -> list[HypothesisItem]:
    """Normalize either writer output shape into a HypothesisItem list."""
    if isinstance(output, HypothesisCandidateSet):
        return [
            HypothesisItem(category=h.category, evidence=h.evidence, hypothesis=h.hypothesis)
            for h in output.hypotheses
        ]
    return [HypothesisItem(hypothesis=output.hypothesis, evidence=output.evidence)]


async def run_baseline(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    profile_name: str,
) -> Submission:
    """Single-pass baseline generation for one BenchmarkCase."""
    writer = base_writer_agent(domain, profile_name=profile_name)
    run = await Runner.run(writer, base_writer_prompt(case.context))
    output = coerce_to_model(
        run.final_output,
        HypothesisCandidateSet if domain.multi_hypothesis else HypothesisCandidate,
    )
    tokens = _extract_usage(run)

    submission = Submission(
        domain=domain.name,
        hypotheses=_candidate_to_items(output),
        id=case.id,
        provenance=GenerationProvenance(
            mode="baseline",
            profile=profile_name,
            tokens=tokens,
        ),
    )
    logger.info("baseline.done id=%s tokens=%d", case.id, tokens.total_tokens)
    return submission


def _normalize_skill_selection(selection: SkillSelection) -> list[str]:
    """Filter to known skills, deduplicate, cap at MAX_SKILLS_PER_CASE."""
    seen: set[str] = set()
    chosen: list[str] = []
    for raw in selection.skills:
        name = raw.strip()
        if not name or name in seen or name not in SKILLS:
            continue
        seen.add(name)
        chosen.append(name)
        if len(chosen) >= MAX_SKILLS_PER_CASE:
            break
    return chosen


async def select_skills(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    profile_name: str,
) -> tuple[list[str], TokenUsage]:
    """Ask the selector LLM to pick 0–3 skills for this case."""
    skills = [SKILLS[name] for name in SKILL_NAMES]
    selector = skill_selector_agent(domain, profile_name=profile_name)
    run = await Runner.run(selector, selector_prompt(case.context, skills))
    raw = coerce_to_model(run.final_output, SkillSelection)
    chosen = _normalize_skill_selection(raw)
    return chosen, _extract_usage(run)


async def _run_agent_pipeline(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    pipeline: list[Skill],
    profile_name: str,
) -> tuple[HypothesisCandidate | HypothesisCandidateSet, TokenUsage]:
    """Execute a sequence of skills; intermediate stages emit prose, the last emits structure."""
    tokens = TokenUsage()
    prior_analyses: list[tuple[str, str]] = []

    for stage_index, skill in enumerate(pipeline[:-1], start=1):
        intermediate = agent_intermediate_writer(
            domain,
            profile_name=profile_name,
            skill_framework=skill.body,
            skill_name=skill.name,
            stage_index=stage_index,
            stage_total=len(pipeline),
        )
        run = await Runner.run(
            intermediate,
            agent_intermediate_prompt(case.context, prior_analyses),
        )
        tokens = tokens + _extract_usage(run)
        analysis = str(run.final_output)
        prior_analyses.append((skill.name, analysis))

    final_skill = pipeline[-1]
    finalizer = agent_final_writer(
        domain,
        profile_name=profile_name,
        skill_framework=final_skill.body,
        skill_name=final_skill.name,
    )
    final_run = await Runner.run(
        finalizer,
        agent_final_prompt(case.context, prior_analyses),
    )
    tokens = tokens + _extract_usage(final_run)
    output = coerce_to_model(
        final_run.final_output,
        HypothesisCandidateSet if domain.multi_hypothesis else HypothesisCandidate,
    )
    return output, tokens


async def run_agent(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    profile_name: str,
) -> Submission:
    """Agent-Mode generation: skill selection (1–3, required) → skill pipeline."""
    selected, selector_tokens = await select_skills(
        domain=domain, case=case, profile_name=profile_name,
    )

    if not selected:
        raise ValueError(
            f"agent selector returned no valid skills for case {case.id!r}; "
            f"Agent Mode requires 1–3 skills drawn from the menu."
        )

    pipeline = [SKILLS[name] for name in selected]
    output, pipeline_tokens = await _run_agent_pipeline(
        case=case, domain=domain, pipeline=pipeline, profile_name=profile_name,
    )

    tokens = selector_tokens + pipeline_tokens
    submission = Submission(
        domain=domain.name,
        hypotheses=_candidate_to_items(output),
        id=case.id,
        provenance=GenerationProvenance(
            mode="agent",
            profile=profile_name,
            skills_used=selected,
            tokens=tokens,
        ),
    )
    logger.info(
        "agent.done id=%s skills=%s tokens=%d",
        case.id, ",".join(selected), tokens.total_tokens,
    )
    return submission


async def generate_submission(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    mode: str,
    profile_name: str,
) -> Submission:
    """Dispatch to baseline or agent mode."""
    if mode == "baseline":
        return await run_baseline(case=case, domain=domain, profile_name=profile_name)
    if mode == "agent":
        return await run_agent(case=case, domain=domain, profile_name=profile_name)
    raise ValueError(f"Unknown generation mode: {mode!r}")


def encode_submission(submission: Submission) -> str:
    """Serialize a Submission to a single JSON line."""
    return json.dumps(submission.model_dump(mode="json"), ensure_ascii=False)
