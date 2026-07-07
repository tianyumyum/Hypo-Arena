"""Score pipeline: per-submission absolute rubric scoring + leaderboard."""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

from agents import Runner

from basics import (
    BenchmarkCase,
    DomainConfig,
    Leaderboard,
    LeaderboardEntry,
    LeaderboardMetadata,
    Pool,
    RecallStats,
    ScoreRecord,
    Submission,
    keep_in_pool,
)

from basics.parsing import coerce_to_model

from .agents import score_judge_agent
from .prompts import score_judge_prompt
from .rubric import compute_q_pair, compute_set_score, compute_summary_score
from .schema import ScoreVerdict

logger = logging.getLogger("hypo.evaluation.score")


def _parse_recall(raw: str | None, *, total_hint: int) -> RecallStats | None:
    """Parse a 'hits/total' string from the judge into RecallStats; clamp hits to [0, total]."""
    if not raw:
        return None
    text = str(raw).strip()
    if "/" not in text:
        return None
    head, tail = text.split("/", 1)
    try:
        hits = int(head.strip())
        total = int(tail.strip())
    except ValueError:
        return None
    if total <= 0:
        return None
    if total != total_hint:
        logger.warning(
            "score.recall: judge reported total=%d but reference has %d hypotheses; using judge's value",
            total, total_hint,
        )
    hits = max(0, min(hits, total))
    return RecallStats(hits=hits, total=total)


async def score_submission(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    judge_profile: str,
    model: str,
    submission: Submission,
    with_recall: bool = False,
) -> ScoreRecord:
    """Score one submission for one case under the rubric; optionally compute reference recall."""
    use_recall = with_recall and domain.multi_hypothesis and bool(case.hypotheses)
    judge = score_judge_agent(domain, profile_name=judge_profile, with_recall=use_recall)
    run = await Runner.run(
        judge,
        score_judge_prompt(
            context=case.context,
            reference=case.hypotheses if use_recall else None,
            submission=submission.hypotheses,
        ),
    )
    raw = coerce_to_model(run.final_output, ScoreVerdict)
    k = len(submission.hypotheses)

    pair_scores = [p.model_dump() for p in raw.pair_scores]
    if k and len(pair_scores) != k:
        logger.warning(
            "score.pair_scores: judge returned %d per-pair entries but Submission has %d Hypotheses",
            len(pair_scores), k,
        )

    set_scores: dict[str, float] = {}
    if domain.multi_hypothesis and raw.set_scores is not None:
        dump = raw.set_scores.model_dump()
        # Paper §3.1.2: distinctness is N/A when only 1 Hypothesis is submitted.
        if k <= 1:
            dump.pop("distinctness", None)
        set_scores = {key: val for key, val in dump.items() if val is not None}

    recall = _parse_recall(raw.recall, total_hint=len(case.hypotheses)) if use_recall else None
    return ScoreRecord(
        case_id=case.id,
        judge=judge_profile,
        model=model,
        pair_scores=pair_scores,
        rationale=raw.rationale,
        recall=recall,
        set_scores=set_scores,
    )


async def score_case(
    *,
    case: BenchmarkCase,
    concurrency: int,
    domain: DomainConfig,
    judge_profile: str,
    submissions: dict[str, Submission],
    with_recall: bool = False,
) -> list[ScoreRecord]:
    """Score every submission for one case in parallel (bounded by `concurrency`)."""
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(model: str, submission: Submission) -> ScoreRecord:
        async with semaphore:
            return await score_submission(
                case=case,
                domain=domain,
                judge_profile=judge_profile,
                model=model,
                submission=submission,
                with_recall=with_recall,
            )

    return await asyncio.gather(
        *(_run_one(model, sub) for model, sub in submissions.items())
    )


def _aggregate_per_model(
    records: list[ScoreRecord],
    *,
    multi: bool,
) -> dict[str, dict[str, float]]:
    """Average pair/set/summary dimensions per model across all cases."""
    pair_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    recall_acc: dict[str, list[float]] = defaultdict(list)
    set_acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    summary_acc: dict[str, list[float]] = defaultdict(list)

    for r in records:
        for pair in r.pair_scores:
            for k, v in pair.items():
                pair_acc[r.model][k].append(v)
        if multi:
            for k, v in r.set_scores.items():
                set_acc[r.model][k].append(v)
        if r.recall is not None:
            recall_acc[r.model].append(r.recall.ratio)
        summary_acc[r.model].append(
            compute_summary_score(
                compute_q_pair(r.pair_scores),
                compute_set_score(r.set_scores) if multi else None,
                multi=multi,
            )
        )

    aggregates: dict[str, dict[str, float]] = {}
    models = set(summary_acc) | set(pair_acc) | set(set_acc) | set(recall_acc)
    for model in models:
        breakdown: dict[str, float] = {}
        for k, vals in pair_acc.get(model, {}).items():
            breakdown[k] = sum(vals) / len(vals)
        if multi:
            for k, vals in set_acc.get(model, {}).items():
                breakdown[k] = sum(vals) / len(vals)
        if recall_acc.get(model):
            vals = recall_acc[model]
            breakdown["recall"] = sum(vals) / len(vals)
        breakdown["S"] = sum(summary_acc[model]) / len(summary_acc[model])
        aggregates[model] = breakdown
    return aggregates


def build_leaderboard(
    *,
    config: str,
    domain: DomainConfig,
    judge_profile: str,
    records: list[ScoreRecord],
) -> Leaderboard:
    """Assemble a score leaderboard from per-(case, model) ScoreRecords."""
    aggregates = _aggregate_per_model(records, multi=domain.multi_hypothesis)
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        counts[r.model] += 1

    ranked = sorted(aggregates.items(), key=lambda kv: -kv[1].get("S", 0.0))
    entries = [
        LeaderboardEntry(
            breakdown={k: round(v, 4) for k, v in breakdown.items()},
            model=model,
            n_observations=counts[model],
            rank=rank,
            rating=round(breakdown.get("S", 0.0), 4),
        )
        for rank, (model, breakdown) in enumerate(ranked, start=1)
    ]
    metadata = LeaderboardMetadata(
        config=config,
        domain=domain.name,
        judge=judge_profile,
        method="score",
        n_models=len(entries),
        n_observations=len(records),
    )
    return Leaderboard(metadata=metadata, rankings=entries)


def filter_records_by_pool(records: list[ScoreRecord], pool: Pool) -> list[ScoreRecord]:
    """Keep only score records whose model belongs to ``pool`` (reference always retained).

    For ``pool='full'`` returns the input list unchanged (no copy); callers must treat
    the result as read-only.
    """
    if pool == "full":
        return records
    return [r for r in records if keep_in_pool(r.model, pool)]


def build_pool_leaderboard(
    *,
    config: str,
    domain: DomainConfig,
    judge_profile: str,
    records: list[ScoreRecord],
    pool: Pool,
) -> Leaderboard:
    """Build a rubric leaderboard restricted to ``pool``; metadata.pool is set accordingly."""
    sub = filter_records_by_pool(records, pool)
    leaderboard = build_leaderboard(
        config=config, domain=domain, judge_profile=judge_profile, records=sub,
    )
    leaderboard.metadata.pool = pool
    return leaderboard


def encode_record(record: ScoreRecord) -> str:
    """Serialize one ScoreRecord to a JSON line."""
    return json.dumps(record.model_dump(mode="json"), ensure_ascii=False)


def encode_leaderboard(leaderboard: Leaderboard) -> str:
    """Serialize a Leaderboard to a single JSON string (indented)."""
    return json.dumps(leaderboard.model_dump(mode="json"), ensure_ascii=False, indent=2)
