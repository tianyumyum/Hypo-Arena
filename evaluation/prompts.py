"""Evaluation prompts: arena (pairwise) judge and score (absolute) judge."""

from __future__ import annotations

import json

from basics import DomainConfig, HypothesisItem

from construction.prompts import BENCHMARK_OVERVIEW, domain_profile, principles_for

from .rubric import (
    PAIR_KEYS,
    SET_KEYS,
    domain_emphasis,
    render_dimensions_block,
    render_recall_block,
)


# ---- shared building blocks ----

def _judge_overview(domain: DomainConfig) -> str:
    profile = domain_profile(domain)
    return (
        f"{BENCHMARK_OVERVIEW}\n\n"
        f"{principles_for(domain)}\n"
        f"You are now at HypoArena's evaluation stage for {profile['setting']}, "
        f"reading the case as {profile['reader_persona']} would.\n"
    )


# ---- arena judge ----

_ARENA_ROLE = """\
You are a HypoArena arena judge. You compare two anonymous Submissions —
'A' and 'B' — written for the same Context, and decide which one is the
better hypothesis-side response under the rubric below.

You return a 5-level verdict and a short rationale that names the
rubric-grounded reason for your call. Do NOT reveal your confidence in
any other form, do NOT score each side numerically, and do NOT critique
the Context — only the Submissions.

Verdict scale (use one of these tokens):
  * "A>>B"  — A is clearly and substantially better
  * "A>B"   — A is better
  * "A=B"   — neither is meaningfully better
  * "B>A"   — B is better
  * "B>>A"  — B is clearly and substantially better

Output format: return ONLY a single JSON object with exactly two keys:
  {{"verdict": "<one of the tokens above>", "rationale": "<your reason>"}}
No markdown fences, no prose before or after the JSON.

Rubric to apply:
{rubric_block}

Domain emphasis (apply on top of the rubric):
{emphasis}

Calibration notes:
  * Length, fluency, and confidence-of-tone are not quality signals on
    their own. Penalize verbose Submissions that pad weak grounding.
  * Speculative specificity (impressive-sounding details the Context
    cannot warrant) is a weakness, not a strength.
  * Stylistic similarity to any familiar reference style is not a
    quality signal — judge analytical depth and Context grounding alone.
  * Avoid position bias: the order of A and B is randomized; judge on
    content alone.
  * For multi-hypothesis Submissions, weigh pair-level quality (each
    Hypothesis on its own merits) and set-level quality (the set as a
    whole — breadth, distinctness, utility) together. A Submission with
    one outstanding Hypothesis but redundant or shallow companions is
    not necessarily better than a smaller, tighter set; nor is a larger
    set automatically broader if the extra entries dilute analytical
    sharpness. Each Submission's Hypotheses carry an explicit
    `[category: ...]` lane label when applicable — use these to assess
    distinctness directly rather than inferring lanes from text alone.
"""


def arena_judge_instructions(domain: DomainConfig) -> str:
    """System instructions for the arena judge."""
    return (
        _judge_overview(domain)
        + "\n"
        + _ARENA_ROLE.format(
            rubric_block=render_dimensions_block(domain),
            emphasis=domain_emphasis(domain),
        )
    )


def _format_hypotheses(hypotheses: list[HypothesisItem]) -> str:
    """Render a hypothesis list as a numbered, indented prose block."""
    if not hypotheses:
        return "(no hypotheses)"
    parts: list[str] = []
    for i, h in enumerate(hypotheses, 1):
        head = f"  ({i})"
        if h.category:
            head += f" [category: {h.category}]"
        parts.append(f"{head} Hypothesis: {h.hypothesis}\n      Evidence: {h.evidence}")
    return "\n".join(parts)


def arena_judge_prompt(
    *,
    context: str,
    submission_a: list[HypothesisItem],
    submission_b: list[HypothesisItem],
) -> str:
    """User prompt for one direction of an arena comparison."""
    return (
        "Read the Context, then compare Submissions A and B and return your "
        "verdict in the schema your instructions describe.\n\n"
        "=== Context ===\n"
        f"{context}\n"
        "=== End Context ===\n\n"
        "=== Submission A ===\n"
        f"{_format_hypotheses(submission_a)}\n"
        "=== End Submission A ===\n\n"
        "=== Submission B ===\n"
        f"{_format_hypotheses(submission_b)}\n"
        "=== End Submission B ===\n"
    )


# ---- score judge ----

_SCORE_ROLE = """\
You are a HypoArena score judge. You read one anonymous Submission
against its Context and assign integer 1–5 scores per Hypothesis on
each rubric dimension below, then add set-level scores when the
Submission contains multiple Hypotheses. Your scores feed a diagnostic
dashboard, not a leaderboard, so calibrate honestly — do not anchor
near the middle when meaningful depth or weakness is present.

Score scale per dimension:
  1 — fails the dimension
  2 — weak
  3 — adequate
  4 — strong
  5 — exemplary

Output format: return ONLY a single JSON object with these top-level keys
(every key MUST be present; use null where N/A). No markdown fences, no prose
outside the JSON.

  * pair_scores: a list of objects, one per Hypothesis in the
    Submission (in the same order they appear in the input). Each
    object has keys {pair_keys} and scores that Hypothesis individually
    on the three pair-level dimensions. Do NOT collapse the list into
    a single average — score each Hypothesis on its own merits;
    aggregation is performed downstream.
  * set_scores: {set_scores_instruction}
  * rationale: 2–4 sentences explaining the most consequential
    strengths and weaknesses you saw, grounded in specific
    Context-evidence links across the Hypotheses.
  * recall: {recall_instruction}

Rubric to apply:
{rubric_block}
{recall_rubric_block}

Domain emphasis (apply on top of the rubric):
{emphasis}

Calibration notes:
  * Length, fluency, and confidence-of-tone are not quality signals on
    their own.
  * Speculative specificity (impressive-sounding details the Context
    cannot warrant) is a weakness, not a strength.
  * Use the full 1–5 range when warranted; do not anchor near 3 by
    default. Each Submission's Hypotheses carry an explicit
    `[category: ...]` lane label when applicable — use these when
    judging set-level distinctness rather than inferring lanes from
    text alone.
  * When the Submission contains only 1 Hypothesis in a domain that
    admits multi-pair Submissions, set `distinctness` to null inside
    set_scores (it is N/A with a single Hypothesis); still report
    numeric values for `breadth` and `utility`.
{recall_calibration_note}"""


def _set_scores_instruction(domain: DomainConfig) -> str:
    if domain.multi_hypothesis:
        return (
            f"object with keys {{{', '.join(SET_KEYS)}}}, all numeric "
            f"(set `distinctness` to null when only 1 Hypothesis is submitted)."
        )
    return "set to null (this domain does not use set-level scoring)."


def _recall_instruction(*, with_recall: bool) -> str:
    if with_recall:
        return (
            "a string \"hits/total\" reporting how many reference "
            "Hypotheses the Submission substantively covers (see rubric)."
        )
    return "set to null (no reference supplied for this case)."


def _recall_rubric_block(*, with_recall: bool) -> str:
    if not with_recall:
        return ""
    return "\n\n" + render_recall_block()


def _recall_calibration_note(*, with_recall: bool) -> str:
    if not with_recall:
        return ""
    return (
        "\n  * For recall, judge core mechanism / conclusion alignment, not "
        "lexical or topical overlap. A reference Hypothesis only counts as "
        "hit if at least one Submission Hypothesis genuinely matches it."
    )


def score_judge_instructions(domain: DomainConfig, *, with_recall: bool = False) -> str:
    """System instructions for the score judge; optionally activates the recall diagnostic."""
    return (
        _judge_overview(domain)
        + "\n"
        + _SCORE_ROLE.format(
            emphasis=domain_emphasis(domain),
            pair_keys="{" + ", ".join(PAIR_KEYS) + "}",
            recall_calibration_note=_recall_calibration_note(with_recall=with_recall),
            recall_instruction=_recall_instruction(with_recall=with_recall),
            recall_rubric_block=_recall_rubric_block(with_recall=with_recall),
            rubric_block=render_dimensions_block(domain),
            set_scores_instruction=_set_scores_instruction(domain),
        )
    )


def score_judge_prompt(
    *,
    context: str,
    reference: list[HypothesisItem] | None = None,
    submission: list[HypothesisItem],
) -> str:
    """User prompt for absolute scoring; supplying reference activates the recall diagnostic."""
    parts = [
        "Read the Context, then score the Submission below using the rubric "
        "in your instructions. Return the result in the schema your "
        "instructions describe.",
        "",
        "=== Context ===",
        context,
        "=== End Context ===",
        "",
        "=== Submission ===",
        _format_hypotheses(submission),
        "=== End Submission ===",
    ]
    if reference is not None:
        parts.extend([
            "",
            "=== Reference (for recall only) ===",
            _format_hypotheses(reference),
            "=== End Reference ===",
        ])
    return "\n".join(parts) + "\n"


# ---- helper for embedding submissions into raw JSON if a caller wants it ----

def submission_as_json(hypotheses: list[HypothesisItem]) -> str:
    """Render hypotheses as a JSON list (for callers that prefer raw structure)."""
    return json.dumps([h.model_dump() for h in hypotheses], ensure_ascii=False, indent=2)
