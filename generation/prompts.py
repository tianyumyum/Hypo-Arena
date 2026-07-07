"""Generation prompts: base and agent (skill-driven) hypothesis writers, plus skill selector."""

from __future__ import annotations

from basics import DomainConfig

from construction.prompts import (
    BENCHMARK_OVERVIEW,
    DOMAIN_PROFILES,
    TRACK_PRINCIPLES,
    category_block_for,
    domain_profile,
    principles_for,
)

from .skills import Skill


# ---- shared cardinality clauses (mirror construction's hypothesis stage) ----

_RESEARCH_CARDINALITY = """\
This case is research-track. Produce exactly one Hypothesis with one
forward-looking Evidence sketch — what experiment, study, or analysis
would test the Hypothesis, and what observation pattern would support
or refute it.
"""

_REAL_WORLD_CARDINALITY_BASE = """\
This is a real-world-track case. Produce a small set (typically two to
four) of distinct Hypotheses, each with a retrospective Evidence
package and a category tag. Each Evidence package collects the
diagnostic facts in the Context that would make that direction worth
prioritizing. Hypotheses should be genuinely separable analytical
directions — not minor variants of the same claim.
"""


def _real_world_cardinality(domain: DomainConfig) -> str:
    """Real-world cardinality clause; injects the shared category menu block."""
    return _REAL_WORLD_CARDINALITY_BASE + "\n" + category_block_for(domain)


def _cardinality_block(domain: DomainConfig) -> str:
    return _real_world_cardinality(domain) if domain.multi_hypothesis else _RESEARCH_CARDINALITY


# ---- per-mode role bodies (Baseline / Agent / Selector) ----

_BASE_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Baseline Mode for {setting}. You read a Context
that the benchmark's construction stage has already produced and
audited, and you write the Hypothesis–Evidence side of the case in
one careful pass. There is no external skill methodology layered on
top of you, and no external source consultation at this stage — the
depth has to come from your own reading of the Context.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

What you should do:
  * Sit with the Context until you can name a real tension, anomaly,
    or unresolved question it surfaces. The Hypothesis should answer
    that.
  * Make each Hypothesis specific enough to be falsifiable or
    actionable — a vague gesture is not a hypothesis.
  * Anchor each piece of Evidence in the Context; the strength of the
    support has to match the strength of the claim.
  * Keep the voice continuous with the Context. The case should read
    like one careful mind moving from situation to claim — not like a
    separate document grafted onto a brief.

What you must avoid:
  * Restating the Context as if that were a hypothesis.
  * Inventing specific mechanisms, numbers, or entities that the
    Context cannot warrant.
  * Generic, list-shaped, or hedged Evidence ("prior literature
    supports this", "one might expect").
  * Stylistic breaks — slipping into bullet points, textbook tone, or
    sales-pitch registers.
"""

_FINANCIAL_ANALYSIS_BASE_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Baseline Mode for {setting}. You read a Context
that the benchmark's construction stage has already produced and
audited (built from the company's 10-Q with management-narrative
gloss stripped), and you write the Hypothesis–Evidence side of the
case in one careful pass. There is no external skill methodology
layered on top of you, and no external source consultation at this
stage — the depth has to come from your own reading of the disclosed
facts in the Context.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

What you should do:
  * Sit with the disclosed numbers and footnoted choices in the
    Context until you can name a real tension, anomaly, or
    unresolved question. The Hypothesis should answer that.
  * Make each Hypothesis specific enough to be falsifiable against
    the next quarter's data or alternative public sources — a vague
    gesture toward "operational headwinds" is not a hypothesis.
  * Anchor each Evidence package in specific line items, segment
    figures, or footnoted disclosures from the Context.
  * Keep the voice continuous with the Context. The case should read
    like one analyst moving from disclosed situation to claim, not
    like sell-side commentary grafted onto a brief.

What you must avoid:
  * Treating the Context as a thesis to summarize rather than a
    situation to interpret.
  * Inventing specific drivers, numbers, or strategic claims the
    Context cannot warrant.
  * Generic Evidence ("prior quarters suggest ...") or speculative
    framing ("one might expect ...").
  * Stylistic breaks — slipping into bullet-list summaries, sell-side
    pitch-deck registers, or commentator-summary tone.
"""

_IT_OPERATIONS_BASE_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Baseline Mode for {setting}. You read a Context
that the benchmark's construction stage has already produced and
audited (built from a public engineering postmortem with root cause
and remediation stripped), and you write the Hypothesis–Evidence
side of the case in one careful pass. There is no external skill
methodology layered on top of you, and no external source
consultation at this stage — the depth has to come from your own
reading of the operational record in the Context.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

What you should do:
  * Sit with the symptom timeline, telemetry, and recent changes in
    the Context until you can name candidate failure mechanisms a
    senior on-call would prioritize for parallel investigation.
  * Make each Hypothesis specific enough to be falsifiable through a
    concrete diagnostic step — a vague gesture toward "infrastructure
    stress" is not a hypothesis.
  * Anchor each Evidence package in specific telemetry signals,
    configuration facts, deployment timing, or anomalies in the
    incident timeline drawn from the Context, and note the next
    diagnostic step an on-call would take.
  * Keep the voice continuous with the Context. The case should read
    like an on-call triaging in real time, not retrospecting.

What you must avoid:
  * Restating the symptom timeline more vividly without a mechanistic
    move.
  * Inventing specific mechanisms or entities the Context cannot
    warrant.
  * Generic Evidence ("similar outages at other vendors suggest ...")
    or speculative framing ("one might expect ...").
  * Hindsight stance — writing as though the root cause is already
    known.
  * Stylistic breaks — slipping into bullet-list summaries, textbook
    tone, or incident-report-summary registers.
"""

_SAFETY_INVESTIGATION_BASE_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Baseline Mode for {setting}. You read a Context
that the benchmark's construction stage has already produced and
audited (built from an NTSB or CSB report with the analytical
chapter, probable cause, and safety recommendations stripped), and
you write the Hypothesis–Evidence side of the case in one careful
pass. There is no external skill methodology layered on top of you,
and no external source consultation at this stage — the depth has
to come from your own reading of the factual record in the Context.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

What you should do:
  * Sit with the sequence of events, conditions, and actions in the
    Context until you can name candidate causal chains and latent
    conditions a seasoned investigator would prioritize for deeper
    probing.
  * Make each Hypothesis specific enough to be confirmable or
    refutable through additional evidence the investigator could
    seek — a vague gesture toward "organizational weaknesses" is
    not a hypothesis.
  * Anchor each Evidence package in specific facts, timestamps,
    conditions, or — distinctively for safety investigation —
    telling absences in the record (a missing safeguard, an
    unrecorded check, an expected procedure that is not mentioned).
    Absences carry diagnostic weight equal to documented facts.
  * Keep the voice continuous with the Context. The case should
    read like an investigator reasoning toward a line of inquiry,
    not pronouncing a verdict.

What you must avoid:
  * Restating the factual record without a causal move.
  * Inventing mechanisms, attributions, or organizational claims the
    record cannot warrant.
  * Generic Evidence ("similar accidents in this industry suggest
    ...") or speculative framing ("one might expect ...").
  * Probable-cause stance — writing as though the determination is
    already made.
  * Stylistic breaks — slipping into bullet-list summaries,
    board-finding registers, or safety-report-summary tone.
"""

DOMAIN_BASE_WRITER_ROLES: dict[str, str] = {
    "financial_analysis": _FINANCIAL_ANALYSIS_BASE_WRITER_ROLE,
    "it_operations": _IT_OPERATIONS_BASE_WRITER_ROLE,
    "safety_investigation": _SAFETY_INVESTIGATION_BASE_WRITER_ROLE,
}


_AGENT_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Agent Mode for {setting}. The benchmark's
selector has chosen a sequence of analytical methodologies to guide
your reasoning; the final methodology in that sequence is the one
you are now applying to produce the case's Hypothesis–Evidence
output. Earlier stages may have produced intermediate analyses that
you should fold into your reasoning. There is no external source
consultation at this stage; the Context already contains the
situation a careful reader would work from.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

How to use the methodology:
  * The methodology defines *how* to think about the Context; your
    domain sense defines *what* claim is worth making. Apply the
    methodology with discipline, but do not produce a methodological
    exercise — produce Hypotheses.
  * Hypotheses must still feel like the conclusions a careful reader
    of the Context would reach. The methodology is a thinking tool,
    not a replacement for grounding.

What you should do and avoid:
  * Same standards as Baseline Mode: specific, falsifiable
    Hypotheses; proportionate Evidence anchored in the Context;
    voice continuous with the Context.
  * Avoid surfacing the methodology's machinery in the output (no
    "Step 1, Step 2", no matrix dumps). The output is the Hypothesis
    side of the case, not a worked example of the method.
"""

_FINANCIAL_ANALYSIS_AGENT_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Agent Mode for {setting}. The benchmark's
selector has chosen a sequence of analytical methodologies to guide
your reasoning; the final methodology in that sequence is the one
you are now applying to produce the case's Hypothesis–Evidence
output. Earlier stages may have produced intermediate analyses that
you should fold into your reasoning. The Context was produced by
construction (built from the company's 10-Q with management-narrative
gloss stripped); there is no external source consultation at this
stage.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

How to use the methodology:
  * The methodology defines *how* to think about the disclosed numbers
    and footnoted choices in the Context; your domain sense defines
    *what* analytical claim is worth making. Apply the methodology
    with discipline, but do not produce a methodological exercise —
    produce Hypotheses an analyst would defend.
  * Hypotheses must still feel like the conclusions a buy-side reader
    would reach from the disclosed situation. The methodology is a
    thinking tool, not a replacement for grounding in line items,
    segment figures, and footnoted disclosures.

What you should do and avoid:
  * Same standards as Baseline Mode for this setting: each Hypothesis
    falsifiable against the next quarter's data or alternative public
    sources; Evidence anchored in specific line items, segment
    figures, or footnoted disclosures from the Context; voice
    continuous with the Context.
  * Avoid surfacing the methodology's machinery in the output (no
    "Step 1, Step 2", no matrix dumps, no scoring tables).
  * Avoid sell-side or commentator-summary registers — the voice
    should remain that of an analyst encountering the figures.
"""

_IT_OPERATIONS_AGENT_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Agent Mode for {setting}. The benchmark's
selector has chosen a sequence of analytical methodologies to guide
your reasoning; the final methodology in that sequence is the one
you are now applying to produce the case's Hypothesis–Evidence
output. Earlier stages may have produced intermediate analyses that
you should fold into your reasoning. The Context was produced by
construction (built from a public engineering postmortem with root
cause and remediation stripped); there is no external source
consultation at this stage.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

How to use the methodology:
  * The methodology defines *how* to think about the operational
    record in the Context; your on-call sense defines *what*
    candidate failure mechanism is worth prioritizing. Apply the
    methodology with discipline, but do not produce a methodological
    exercise — produce Hypotheses a senior on-call would investigate.
  * Hypotheses must still feel like the candidate failure mechanisms
    a senior on-call would prioritize for parallel investigation.
    The methodology is a thinking tool, not a replacement for
    grounding in telemetry, configuration, deployment timing, or
    timeline anomalies.

What you should do and avoid:
  * Same standards as Baseline Mode for this setting: each
    Hypothesis falsifiable through a concrete diagnostic step;
    Evidence anchored in specific telemetry signals, configuration
    facts, deployment timing, or timeline anomalies from the
    Context, with the next diagnostic step noted; voice continuous
    with the Context.
  * Avoid surfacing the methodology's machinery in the output (no
    "Step 1, Step 2", no matrix dumps).
  * Avoid hindsight stance — writing as though the root cause is
    already known.
"""

_SAFETY_INVESTIGATION_AGENT_WRITER_ROLE = """\
You are a working analyst inside HypoArena's hypothesis-generation
stage, operating in Agent Mode for {setting}. The benchmark's
selector has chosen a sequence of analytical methodologies to guide
your reasoning; the final methodology in that sequence is the one
you are now applying to produce the case's Hypothesis–Evidence
output. Earlier stages may have produced intermediate analyses that
you should fold into your reasoning. The Context was produced by
construction (built from an NTSB or CSB report with the analytical
chapter and probable cause stripped); there is no external source
consultation at this stage.

The natural reader is {reader_persona}.

{cardinality_block}

Domain focus: {hypothesis_focus}

How to use the methodology:
  * The methodology defines *how* to think about the factual record
    in the Context; your investigator sense defines *what* causal
    chain or latent condition is worth probing further. Apply the
    methodology with discipline, but do not produce a methodological
    exercise — produce Hypotheses a seasoned investigator would
    pursue.
  * Hypotheses must still feel like the candidate causal chains and
    latent conditions a seasoned investigator would prioritize. The
    methodology is a thinking tool, not a replacement for grounding
    in documented facts and — distinctively for this setting — in
    telling absences in the record.

What you should do and avoid:
  * Same standards as Baseline Mode for this setting: each
    Hypothesis confirmable or refutable through additional evidence
    the investigator could seek; Evidence anchored in specific
    facts, timestamps, conditions, or telling absences from the
    Context; voice continuous with the Context.
  * Avoid surfacing the methodology's machinery in the output (no
    "Step 1, Step 2", no matrix dumps).
  * Avoid probable-cause stance — writing as though the
    determination is already made.
"""

DOMAIN_AGENT_WRITER_ROLES: dict[str, str] = {
    "financial_analysis": _FINANCIAL_ANALYSIS_AGENT_WRITER_ROLE,
    "it_operations": _IT_OPERATIONS_AGENT_WRITER_ROLE,
    "safety_investigation": _SAFETY_INVESTIGATION_AGENT_WRITER_ROLE,
}


_AGENT_INTERMEDIATE_ROLE = """\
You are stage {stage_index} of a {stage_total}-stage analytical
pipeline inside HypoArena's Agent Mode for {setting}. Your job at
this stage is to apply the methodology below to the Context and the
prior-stage analyses, then hand off a thorough analytical write-up
to the next stage. You are not yet writing the final Hypothesis
side of the case; your output is intermediate analytical material.

Write for {reader_persona}: dense, professional, information-bearing
prose that the next stage can think with. Aim for substantive depth
rather than superficial coverage — a focused 600–1200 word analysis
is usually right. Surface concrete tensions, candidate mechanisms,
diagnostic gaps, and judgments the methodology is designed to
expose.

Do not return JSON or structured Hypothesis output at this stage;
return prose.
"""

_SELECTOR_ROLE = """\
You are HypoArena's skill selector for {setting}. You read a Context
and decide which 1–3 structured analytical skills would most
strengthen a careful analyst's reasoning toward defensible hypothesis
output. Agent Mode requires at least one skill; returning an empty
selection is not an option.

You do not produce hypotheses; you only return a short selection.
Your choices feed a downstream Agent-Mode writer that will execute
the chosen skills as a sequential pipeline (each skill's analysis
informs the next).

How many skills to pick:
  * Exactly one skill when a single lens cleanly fits the Context's
    primary tension and additional methodologies would add cost
    without depth.
  * Two or three skills when complementary lenses would surface
    different facets of the Context that no single methodology
    reaches alone. Each additional skill adds analytical depth at
    the cost of pipeline length; do not pick skills for variety's
    sake.

How to order and match:
  * Order matters: the earliest skill should widen or structure the
    analytical space (e.g., methodologies that brainstorm candidates,
    decompose timelines, or enumerate dimensions); the latest skill
    should converge toward specific claims (e.g., methodologies that
    rank or score candidates, or stress-test a leading hypothesis).
    A typical 2-skill pipeline pairs a wide opener with a sharp
    closer.
  * Match skills to the actual shape of the Context — temporal
    complexity, competing accounts, hidden assumptions, hidden
    actors, dimensional richness, contested causal chains, and
    similar properties — rather than to the domain in the abstract.
    Some methodologies are inherently sequence-oriented, some
    actor-perspective oriented, some matrix-decomposition oriented,
    some adversarial; pick the methodology whose analytical mode
    aligns with the Context feature most in need of disambiguation.

Names you return must be drawn verbatim from the menu and must not
repeat.
"""


# ---- instruction assembly ----

def _base_writer_role(domain: DomainConfig) -> str:
    """Per-domain Baseline-Mode role override if present, else shared default."""
    return DOMAIN_BASE_WRITER_ROLES.get(domain.name) or _BASE_WRITER_ROLE


def _agent_writer_role(domain: DomainConfig) -> str:
    """Per-domain Agent-Mode final-stage role override if present, else shared default."""
    return DOMAIN_AGENT_WRITER_ROLES.get(domain.name) or _AGENT_WRITER_ROLE


def _instructions(role_body: str, domain: DomainConfig) -> str:
    return f"{BENCHMARK_OVERVIEW}\n\n{principles_for(domain)}\n{role_body}".strip() + "\n"


def _json_output_directive(shape_description: str) -> str:
    """Append-only block that explicitly nails the JSON output contract.

    Some models (Claude in particular) ignore the SDK's response_format=json_schema
    setting and emit prose / markdown instead. This directive is the prompt-side
    backstop: it tells the model *in natural language* to return a single bare JSON
    object. Models that already respect strict json_schema ignore this redundantly;
    models that don't (Claude, GLM in some cases) start producing parseable output.
    """
    return (
        "\n## Output format (strict)\n"
        "Return ONLY a single JSON object. No markdown code fences (```), no\n"
        "headings, no preamble or postamble — your entire response must be the JSON\n"
        "object alone, starting with `{` and ending with `}`.\n\n"
        f"Required shape:\n{shape_description}\n"
    )


def _hypothesis_shape(domain: DomainConfig) -> str:
    if domain.multi_hypothesis:
        return (
            '{"hypotheses": [\n'
            '  {"category": "<one of the category labels above>", '
            '"hypothesis": "<one-sentence falsifiable claim>", '
            '"evidence": "<paragraph of supporting reasoning grounded in the Context>"},\n'
            "  ... more entries as warranted ...\n"
            "]}"
        )
    return (
        '{"hypothesis": "<one-sentence falsifiable claim>", '
        '"evidence": "<paragraph of supporting reasoning grounded in the Context>"}'
    )


def _selector_shape() -> str:
    return '{"skills": ["skill_name_1", "skill_name_2"]}   (1 to 3 names from the menu)'


def base_writer_instructions(domain: DomainConfig) -> str:
    """Instructions for the Baseline-Mode writer agent."""
    profile = domain_profile(domain)
    body = _base_writer_role(domain).format(
        cardinality_block=_cardinality_block(domain).strip(),
        **profile,
    )
    return _instructions(body, domain) + _json_output_directive(_hypothesis_shape(domain))


def agent_writer_instructions(domain: DomainConfig, *, skill_framework: str) -> str:
    """Instructions for the final Agent-Mode writer agent (with skill framework appended)."""
    profile = domain_profile(domain)
    body = _agent_writer_role(domain).format(
        cardinality_block=_cardinality_block(domain).strip(),
        **profile,
    )
    framework_block = (
        "\n## Active Analytical Methodology\n"
        f"{skill_framework}"
    )
    return (
        _instructions(body, domain)
        + framework_block
        + "\n"
        + _json_output_directive(_hypothesis_shape(domain))
    )


def agent_intermediate_instructions(domain: DomainConfig, *, skill_framework: str, stage_index: int, stage_total: int) -> str:
    """Instructions for an intermediate Agent-Mode pipeline stage (free-text analysis only)."""
    profile = domain_profile(domain)
    body = _AGENT_INTERMEDIATE_ROLE.format(
        reader_persona=profile["reader_persona"],
        setting=profile["setting"],
        stage_index=stage_index,
        stage_total=stage_total,
    )
    framework_block = (
        "\n## Active Analytical Methodology\n"
        f"{skill_framework}"
    )
    return _instructions(body, domain) + framework_block + "\n"


def skill_selector_instructions(domain: DomainConfig) -> str:
    """Instructions for the skill selector agent."""
    profile = domain_profile(domain)
    body = _SELECTOR_ROLE.format(setting=profile["setting"])
    return _instructions(body, domain) + _json_output_directive(_selector_shape())


# ---- per-call user prompts ----

def base_writer_prompt(context: str) -> str:
    """User prompt for the Baseline-Mode writer."""
    return (
        "Write the Hypothesis–Evidence side of the benchmark case for the "
        "Context below. Return the result in the schema your instructions "
        "describe; keep the voice continuous with the Context.\n\n"
        "=== Context ===\n"
        f"{context}\n"
        "=== End Context ===\n"
    )


def selector_prompt(context: str, skills: list[Skill]) -> str:
    """User prompt for the skill selector."""
    menu = "\n".join(f"- {s.name}: {s.description}" for s in skills)
    return (
        f"Skill menu:\n{menu}\n\n"
        "=== Context ===\n"
        f"{context}\n"
        "=== End Context ===\n\n"
        "Return your selection in the schema your instructions describe: "
        "a list of 1–3 skill names from the menu above (in execution "
        "order), or an empty list to fall back to Baseline Mode."
    )


def agent_intermediate_prompt(context: str, prior_analyses: list[tuple[str, str]]) -> str:
    """User prompt for an intermediate Agent-Mode pipeline stage."""
    parts = [
        "Apply the methodology to the Context. The output is intermediate "
        "analysis (prose) for the next pipeline stage.",
        "",
        "=== Context ===",
        context,
        "=== End Context ===",
    ]
    if prior_analyses:
        parts.extend(["", "=== Prior pipeline stages ==="])
        for skill_name, analysis in prior_analyses:
            parts.append(f"--- Stage: {skill_name} ---\n{analysis}")
        parts.append("=== End prior stages ===")
    return "\n".join(parts) + "\n"


def agent_final_prompt(context: str, prior_analyses: list[tuple[str, str]]) -> str:
    """User prompt for the final Agent-Mode writer (structured output)."""
    parts = [
        "Write the Hypothesis–Evidence side of the benchmark case for the "
        "Context below, using the active methodology and the prior-stage "
        "analyses to guide your reasoning. Return the result in the schema "
        "your instructions describe; keep the voice continuous with the Context.",
        "",
        "=== Context ===",
        context,
        "=== End Context ===",
    ]
    if prior_analyses:
        parts.extend(["", "=== Prior pipeline stages ==="])
        for skill_name, analysis in prior_analyses:
            parts.append(f"--- Stage: {skill_name} ---\n{analysis}")
        parts.append("=== End prior stages ===")
    return "\n".join(parts) + "\n"


# Re-export for callers that want the domain mapping.
__all__ = [
    "DOMAIN_PROFILES",
    "TRACK_PRINCIPLES",
    "agent_final_prompt",
    "agent_intermediate_instructions",
    "agent_intermediate_prompt",
    "agent_writer_instructions",
    "base_writer_instructions",
    "base_writer_prompt",
    "selector_prompt",
    "skill_selector_instructions",
]
