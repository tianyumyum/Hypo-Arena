"""Evaluation CLI: arena (full pairwise + BTD) and/or score (rubric) for a (domain, config)."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import configure_runtime
from basics import ALL_DOMAINS, BenchmarkCase, Submission, get_domain
from basics.io import (
    append_arena_match,
    append_score_record,
    existing_arena_pair_keys,
    existing_score_keys,
    load_all_submissions,
    load_arena_matches,
    load_cases,
    load_score_records,
)
from basics.paths import arena_pool_leaderboard_path, score_pool_leaderboard_path
from evaluation import (
    build_arena_pool_leaderboard,
    build_score_pool_leaderboard,
    judge_pair,
    reference_submission,
    score_submission,
)
from evaluation.markdown import render_leaderboard_md
from orchestrator.atomic import write_json_atomic, write_text_atomic

logger = logging.getLogger("hypo.evaluate")


def _gather_submissions(
    *,
    cases: list[BenchmarkCase],
    construction_profile: str,
    domain_name: str,
    models: set[str] | None = None,
    modes: tuple[str, ...] = ("baseline", "agent"),
) -> dict[str, dict[str, Submission]]:
    """Per-case {model_key: Submission} where model_key is e.g. 'baseline:gpt-5.4-high'.

    If ``models`` is given, only submissions whose profile (the part after the
    'mode:' prefix) is in that set are kept; the synthetic 'reference' entry is
    always retained. ``modes`` restricts which generation modes are pooled.
    """
    per_case: dict[str, dict[str, Submission]] = {c.id: {} for c in cases}
    for mode in modes:
        for profile, subs in load_all_submissions(domain_name, mode, construction_profile).items():
            if models is not None and profile not in models:
                continue
            label = f"{mode}:{profile}"
            for sub in subs:
                if sub.id in per_case:
                    per_case[sub.id][label] = sub
    for case in cases:
        per_case[case.id]["reference"] = reference_submission(case)
    return per_case


# ---- arena ----

async def _arena_one_pair(
    *,
    case: BenchmarkCase,
    construction_profile: str,
    domain_name: str,
    judge_profile: str,
    model_a: str,
    model_b: str,
    semaphore: asyncio.Semaphore,
    submissions: dict[str, Submission],
) -> None:
    domain = get_domain(domain_name)
    async with semaphore:
        try:
            match = await judge_pair(
                case=case,
                domain=domain,
                judge_profile=judge_profile,
                model_a=model_a,
                model_b=model_b,
                submission_a=submissions[model_a],
                submission_b=submissions[model_b],
            )
        except Exception as exc:
            logger.exception(
                "arena.fail id=%s pair=%s vs %s err=%s",
                case.id, model_a, model_b, exc,
            )
            return
        append_arena_match(domain_name, construction_profile, judge_profile, match)
        logger.info(
            "arena.ok id=%s a=%s b=%s winner=%s consistent=%s",
            case.id, model_a, model_b, match.forward.winner, match.consistent,
        )


async def run_arena(
    *,
    cases: list[BenchmarkCase],
    concurrency: int,
    construction_profile: str,
    domain_name: str,
    judge_profile: str,
    max_pairs_per_case: int | None,
    seed: int | None,
    submissions_per_case: dict[str, dict[str, Submission]],
) -> None:
    """Round-robin (optionally sampled) for every case, with bidirectional judging.

    When *max_pairs_per_case* is set, each case samples at most that many
    unordered pairs instead of exhaustive C(n,2). BTD converges well with
    ~20 pairs/case even for 14 models.
    """
    import random
    from itertools import combinations

    rng = random.Random(seed)
    done = existing_arena_pair_keys(domain_name, construction_profile, judge_profile)
    pending: list[tuple[BenchmarkCase, str, str]] = []
    for case in cases:
        models = sorted(submissions_per_case.get(case.id, {}))
        all_pairs = list(combinations(models, 2))
        if max_pairs_per_case is not None and len(all_pairs) > max_pairs_per_case:
            all_pairs = rng.sample(all_pairs, max_pairs_per_case)
        for a, b in all_pairs:
            if rng.random() < 0.5:
                a, b = b, a
            if (case.id, a, b) in done or (case.id, b, a) in done:
                continue
            pending.append((case, a, b))

    logger.info(
        "arena.start domain=%s judge=%s pending_pairs=%d",
        domain_name, judge_profile, len(pending),
    )
    if not pending:
        return
    semaphore = asyncio.Semaphore(concurrency)
    await asyncio.gather(
        *(
            _arena_one_pair(
                case=case,
                construction_profile=construction_profile,
                domain_name=domain_name,
                judge_profile=judge_profile,
                model_a=a,
                model_b=b,
                semaphore=semaphore,
                submissions=submissions_per_case[case.id],
            )
            for case, a, b in pending
        )
    )


_POOLS: tuple[str, ...] = ("baseline", "agent", "full")


def finalize_arena(
    *,
    construction_profile: str,
    domain_name: str,
    judge_profile: str,
) -> None:
    """Aggregate matches into 3 pool BTD leaderboards (baseline / agent / full) and persist them."""
    matches = load_arena_matches(domain_name, construction_profile, judge_profile)
    if not matches:
        logger.warning("arena.finalize: no matches found, skipping leaderboard")
        return
    domain_cfg = get_domain(domain_name)
    for pool in _POOLS:
        leaderboard = build_arena_pool_leaderboard(
            config=construction_profile,
            domain=domain_cfg,
            judge_profile=judge_profile,
            matches=matches,
            pool=pool,
        )
        write_json_atomic(
            arena_pool_leaderboard_path(domain_name, construction_profile, judge_profile, pool),
            leaderboard,
        )
        write_text_atomic(
            arena_pool_leaderboard_path(
                domain_name, construction_profile, judge_profile, pool, suffix="md",
            ),
            render_leaderboard_md(leaderboard),
        )
        logger.info(
            "arena.leaderboard domain=%s judge=%s pool=%s n_models=%d top=%s",
            domain_name, judge_profile, pool, len(leaderboard.rankings),
            leaderboard.rankings[0].model if leaderboard.rankings else "-",
        )


# ---- score ----

async def _score_one(
    *,
    case: BenchmarkCase,
    construction_profile: str,
    domain_name: str,
    judge_profile: str,
    model: str,
    semaphore: asyncio.Semaphore,
    submission: Submission,
    with_recall: bool,
) -> None:
    domain = get_domain(domain_name)
    async with semaphore:
        try:
            record = await score_submission(
                case=case,
                domain=domain,
                judge_profile=judge_profile,
                model=model,
                submission=submission,
                with_recall=with_recall,
            )
        except Exception as exc:
            logger.exception(
                "score.fail id=%s model=%s err=%s", case.id, model, exc,
            )
            return
        append_score_record(domain_name, construction_profile, judge_profile, record)
        recall_tag = (
            f" recall={record.recall.hits}/{record.recall.total}"
            if record.recall is not None else ""
        )
        logger.info(
            "score.ok id=%s model=%s S=%.2f%s",
            case.id, model, record.overall_score, recall_tag,
        )


async def run_score(
    *,
    cases: list[BenchmarkCase],
    concurrency: int,
    construction_profile: str,
    domain_name: str,
    judge_profile: str,
    submissions_per_case: dict[str, dict[str, Submission]],
    with_recall: bool,
) -> None:
    """Score every (case, model) pair not already scored."""
    done = existing_score_keys(domain_name, construction_profile, judge_profile)
    pending: list[tuple[BenchmarkCase, str, Submission]] = []
    for case in cases:
        for model, sub in submissions_per_case.get(case.id, {}).items():
            if (case.id, model) in done:
                continue
            pending.append((case, model, sub))

    logger.info(
        "score.start domain=%s judge=%s pending=%d with_recall=%s",
        domain_name, judge_profile, len(pending), with_recall,
    )
    if not pending:
        return
    semaphore = asyncio.Semaphore(concurrency)
    await asyncio.gather(
        *(
            _score_one(
                case=case,
                construction_profile=construction_profile,
                domain_name=domain_name,
                judge_profile=judge_profile,
                model=model,
                semaphore=semaphore,
                submission=sub,
                with_recall=with_recall,
            )
            for case, model, sub in pending
        )
    )


def finalize_score(
    *,
    construction_profile: str,
    domain_name: str,
    judge_profile: str,
) -> None:
    """Aggregate score records into 3 pool leaderboards and persist them."""
    records = load_score_records(domain_name, construction_profile, judge_profile)
    if not records:
        logger.warning("score.finalize: no records found, skipping leaderboard")
        return
    domain_cfg = get_domain(domain_name)
    for pool in _POOLS:
        leaderboard = build_score_pool_leaderboard(
            config=construction_profile,
            domain=domain_cfg,
            judge_profile=judge_profile,
            records=records,
            pool=pool,
        )
        write_json_atomic(
            score_pool_leaderboard_path(domain_name, construction_profile, judge_profile, pool),
            leaderboard,
        )
        write_text_atomic(
            score_pool_leaderboard_path(
                domain_name, construction_profile, judge_profile, pool, suffix="md",
            ),
            render_leaderboard_md(leaderboard),
        )
        logger.info(
            "score.leaderboard domain=%s judge=%s pool=%s n_models=%d top=%s",
            domain_name, judge_profile, pool, len(leaderboard.rankings),
            leaderboard.rankings[0].model if leaderboard.rankings else "-",
        )


# ---- entrypoint ----

async def run(
    *,
    concurrency: int,
    construction_profile: str,
    domain_name: str,
    judge_profile: str,
    limit: int | None,
    max_pairs_per_case: int | None,
    method: str,
    models: set[str] | None,
    modes: tuple[str, ...],
    seed: int | None,
    with_recall: bool,
) -> None:
    cases = load_cases(domain_name, construction_profile, only_passed=True)
    if limit is not None:
        cases = cases[:limit]
    submissions_per_case = _gather_submissions(
        cases=cases,
        construction_profile=construction_profile,
        domain_name=domain_name,
        models=models,
        modes=modes,
    )

    if method in ("arena", "both"):
        await run_arena(
            cases=cases,
            concurrency=concurrency,
            construction_profile=construction_profile,
            domain_name=domain_name,
            judge_profile=judge_profile,
            max_pairs_per_case=max_pairs_per_case,
            seed=seed,
            submissions_per_case=submissions_per_case,
        )
        finalize_arena(
            construction_profile=construction_profile,
            domain_name=domain_name,
            judge_profile=judge_profile,
        )

    if method in ("score", "both"):
        await run_score(
            cases=cases,
            concurrency=concurrency,
            construction_profile=construction_profile,
            domain_name=domain_name,
            judge_profile=judge_profile,
            submissions_per_case=submissions_per_case,
            with_recall=with_recall,
        )
        finalize_score(
            construction_profile=construction_profile,
            domain_name=domain_name,
            judge_profile=judge_profile,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HypoArena evaluation pipeline")
    parser.add_argument("--domain", choices=list(ALL_DOMAINS), required=True)
    parser.add_argument("--construction-profile", default="gpt-5.4-high")
    parser.add_argument("--judge", default="mimo-v2-pro")
    parser.add_argument("--method", choices=["arena", "score", "both"], default="both")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--models", default=None,
        help="Comma-separated generation profile names to restrict arena/score to "
             "(e.g. 'gpt-5.5-high,claude-opus-4.8-high'). 'reference' is always kept. "
             "Omit to include every model with submissions.",
    )
    parser.add_argument(
        "--modes", default="baseline,agent",
        help="Comma-separated generation modes to pool (subset of 'baseline,agent'). "
             "Use 'baseline' alone for a clean single-pool leaderboard.",
    )
    parser.add_argument(
        "--max-pairs-per-case", type=int, default=None,
        help="Max unordered pairs to sample per case in arena mode. "
             "None (default) = exhaustive C(n,2). E.g. 20 keeps ~20 random pairs/case.",
    )
    parser.add_argument(
        "--with-recall", action="store_true",
        help="Activate the reference-anchored recall diagnostic (multi-hypothesis domains only).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    models = (
        {m.strip() for m in args.models.split(",") if m.strip()}
        if args.models else None
    )
    modes = tuple(m.strip() for m in args.modes.split(",") if m.strip())

    configure_runtime()
    asyncio.run(
        run(
            concurrency=args.concurrency,
            construction_profile=args.construction_profile,
            domain_name=args.domain,
            judge_profile=args.judge,
            limit=args.limit,
            max_pairs_per_case=args.max_pairs_per_case,
            method=args.method,
            models=models,
            modes=modes,
            seed=args.seed,
            with_recall=args.with_recall,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
