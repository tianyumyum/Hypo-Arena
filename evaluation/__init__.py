"""Evaluation stage: arena pairwise judging (BTD) + rubric-based score diagnostics."""

from basics import BenchmarkCase, GenerationProvenance, Submission

from . import arena, score
from .agents import arena_judge_agent, score_judge_agent
from .arena import (
    VERDICT_SCORE,
    build_leaderboard as build_arena_leaderboard,
    build_pool_leaderboard as build_arena_pool_leaderboard,
    compute_btd,
    encode_leaderboard as encode_arena_leaderboard,
    encode_match,
    filter_matches_by_pool,
    judge_case,
    judge_pair,
)
from .markdown import render_leaderboard_md
from .rubric import (
    DOMAIN_EMPHASIS,
    PAIR_DIMENSIONS,
    PAIR_KEYS,
    RECALL_DIMENSION,
    SET_DIMENSIONS,
    SET_KEYS,
    compute_pair_score,
    compute_set_score,
    compute_summary_score,
    domain_emphasis,
    render_dimensions_block,
    render_recall_block,
)
from .schema import ArenaToken, ArenaVerdict, ScoreVerdict
from .score import (
    build_leaderboard as build_score_leaderboard,
    build_pool_leaderboard as build_score_pool_leaderboard,
    encode_leaderboard as encode_score_leaderboard,
    encode_record,
    filter_records_by_pool,
    score_case,
    score_submission,
)


def reference_submission(case: BenchmarkCase) -> Submission:
    """Wrap a BenchmarkCase's reference hypotheses as a Submission for arena/score evaluation."""
    return Submission(
        domain=case.domain,
        hypotheses=case.hypotheses,
        id=case.id,
        provenance=GenerationProvenance(
            mode="baseline",
            profile=f"reference[{case.provenance.profile}]",
        ),
    )


__all__ = [
    "ArenaToken",
    "ArenaVerdict",
    "DOMAIN_EMPHASIS",
    "PAIR_DIMENSIONS",
    "PAIR_KEYS",
    "RECALL_DIMENSION",
    "SET_DIMENSIONS",
    "SET_KEYS",
    "ScoreVerdict",
    "VERDICT_SCORE",
    "arena",
    "arena_judge_agent",
    "build_arena_leaderboard",
    "build_arena_pool_leaderboard",
    "build_score_leaderboard",
    "build_score_pool_leaderboard",
    "compute_btd",
    "compute_pair_score",
    "compute_set_score",
    "compute_summary_score",
    "domain_emphasis",
    "encode_arena_leaderboard",
    "encode_match",
    "encode_record",
    "encode_score_leaderboard",
    "filter_matches_by_pool",
    "filter_records_by_pool",
    "judge_case",
    "judge_pair",
    "reference_submission",
    "render_dimensions_block",
    "render_leaderboard_md",
    "render_recall_block",
    "score",
    "score_case",
    "score_judge_agent",
    "score_submission",
]
