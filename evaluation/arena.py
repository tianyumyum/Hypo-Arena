"""Arena pipeline: full round-robin pairwise judging + BTD aggregation + leaderboard."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
from collections import defaultdict
from itertools import combinations
from typing import Literal

from agents import Runner

from basics import (
    ArenaMatch,
    BenchmarkCase,
    DomainConfig,
    JudgeVerdict,
    Leaderboard,
    LeaderboardEntry,
    LeaderboardMetadata,
    Pool,
    Submission,
    keep_in_pool,
)

from .agents import arena_judge_agent
from .prompts import arena_judge_prompt
from basics.parsing import coerce_to_model

from .schema import ArenaVerdict

logger = logging.getLogger("hypo.evaluation.arena")


VERDICT_SCORE: dict[str, float] = {
    "A>>B": 1.00,
    "A>B":  0.75,
    "A=B":  0.50,
    "B>A":  0.25,
    "B>>A": 0.00,
}


def _verdict_to_winner(verdict: str) -> Literal["a", "b", "tie"]:
    """Collapse the 5-level token to a 3-way winner."""
    if verdict in ("A>>B", "A>B"):
        return "a"
    if verdict in ("B>>A", "B>A"):
        return "b"
    return "tie"


def _to_judge_verdict(raw: ArenaVerdict) -> JudgeVerdict:
    """Convert the LLM's ArenaVerdict into basics.JudgeVerdict (score + winner derived)."""
    return JudgeVerdict(
        rationale=raw.rationale,
        score=VERDICT_SCORE[raw.verdict],
        winner=_verdict_to_winner(raw.verdict),
    )


async def _judge_once(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    judge_profile: str,
    submission_a: Submission,
    submission_b: Submission,
) -> JudgeVerdict:
    """One-direction arena call: submission_a is shown as 'A', submission_b as 'B'."""
    judge = arena_judge_agent(domain, profile_name=judge_profile)
    run = await Runner.run(
        judge,
        arena_judge_prompt(
            context=case.context,
            submission_a=submission_a.hypotheses,
            submission_b=submission_b.hypotheses,
        ),
    )
    raw = coerce_to_model(run.final_output, ArenaVerdict)
    return _to_judge_verdict(raw)


async def judge_pair(
    *,
    case: BenchmarkCase,
    domain: DomainConfig,
    judge_profile: str,
    model_a: str,
    model_b: str,
    submission_a: Submission,
    submission_b: Submission,
) -> ArenaMatch:
    """Bidirectional judging (forward + reverse) for one pair on one case."""
    forward, reverse = await asyncio.gather(
        _judge_once(
            case=case, domain=domain, judge_profile=judge_profile,
            submission_a=submission_a, submission_b=submission_b,
        ),
        _judge_once(
            case=case, domain=domain, judge_profile=judge_profile,
            submission_a=submission_b, submission_b=submission_a,
        ),
    )
    return ArenaMatch(
        case_id=case.id,
        forward=forward,
        judge=judge_profile,
        model_a=model_a,
        model_b=model_b,
        reverse=reverse,
    )


def _enumerate_pairs(model_keys: list[str], *, seed: int | None) -> list[tuple[str, str]]:
    """Every unordered pair of models, with position randomized to spread position bias."""
    rng = random.Random(seed)
    pairs = list(combinations(sorted(model_keys), 2))
    return [(b, a) if rng.random() < 0.5 else (a, b) for a, b in pairs]


async def judge_case(
    *,
    case: BenchmarkCase,
    concurrency: int,
    domain: DomainConfig,
    judge_profile: str,
    seed: int | None,
    submissions: dict[str, Submission],
) -> list[ArenaMatch]:
    """Full round-robin across all provided submissions for one case."""
    pairs = _enumerate_pairs(list(submissions), seed=seed)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def _run_one(a: str, b: str) -> ArenaMatch:
        async with semaphore:
            return await judge_pair(
                case=case,
                domain=domain,
                judge_profile=judge_profile,
                model_a=a,
                model_b=b,
                submission_a=submissions[a],
                submission_b=submissions[b],
            )

    return await asyncio.gather(*(_run_one(a, b) for a, b in pairs))


# ---- BTD (Bradley–Terry–Davidson) aggregation ----

def compute_btd(
    matches: list[ArenaMatch],
    *,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[str, float]:
    """BTD log-scale ratings (ELO-like, centered at 1500)."""
    players = sorted({m.model_a for m in matches} | {m.model_b for m in matches})
    if len(players) < 2:
        return {p: 1500.0 for p in players}

    idx = {p: i for i, p in enumerate(players)}
    n = len(players)
    wins = [[0.0] * n for _ in range(n)]
    ties = [[0.0] * n for _ in range(n)]

    for match in matches:
        i, j = idx[match.model_a], idx[match.model_b]
        s = match.debiased_score
        if s > 0.5:
            wins[i][j] += (s - 0.5) * 2
            ties[i][j] += (1.0 - s) * 2
            ties[j][i] += (1.0 - s) * 2
        elif s < 0.5:
            wins[j][i] += (0.5 - s) * 2
            ties[i][j] += s * 2
            ties[j][i] += s * 2
        else:
            ties[i][j] += 1.0
            ties[j][i] += 1.0

    total_decisive = sum(wins[i][j] for i in range(n) for j in range(n))
    total_ties = sum(ties[i][j] for i in range(n) for j in range(i + 1, n))
    if total_ties > 0 and total_decisive + total_ties > 0:
        tie_frac = total_ties / (total_decisive + total_ties)
        theta = 1.0 + 2.0 * tie_frac
    else:
        theta = 1.5

    gamma = [1.0] * n
    gamma_floor = 1e-4

    converged = False
    iteration = 0
    for iteration in range(1, max_iter + 1):
        gamma_old = list(gamma)
        theta_old = theta

        for i in range(n):
            num = 0.0
            den = 0.0
            for j in range(n):
                if i == j:
                    continue
                n_ij = wins[i][j] + wins[j][i] + ties[i][j]
                if n_ij == 0:
                    continue
                num += wins[i][j] + 0.5 * ties[i][j]
                den += (
                    n_ij * (theta * gamma[j]) / (gamma[i] + theta * gamma[j])
                    + n_ij * gamma[j] / (theta * gamma[i] + gamma[j])
                ) / 2.0
            if den > 0:
                gamma[i] = max(num / den, gamma_floor)

        log_mean = sum(math.log(g + 1e-15) for g in gamma) / n
        scale = math.exp(log_mean)
        if scale > 0:
            gamma = [g / scale for g in gamma]

        theta_num = 0.0
        theta_den = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                n_ij = wins[i][j] + wins[j][i] + ties[i][j]
                if n_ij == 0:
                    continue
                theta_num += ties[i][j]
                sqrt_pij = math.sqrt(gamma[i] * gamma[j]) if gamma[i] > 0 and gamma[j] > 0 else 0.0
                theta_den += n_ij * 2.0 * sqrt_pij / (
                    gamma[i] + theta * gamma[j] + theta * gamma[i] + gamma[j]
                )
        if theta_den > 0:
            theta = max(1e-4, theta_num / theta_den)

        max_diff = max(abs(gamma[i] - gamma_old[i]) for i in range(n))
        if max_diff < tol and abs(theta - theta_old) < tol:
            converged = True
            break

    if not converged:
        logger.warning(
            "BTD did not converge after %d iterations (n_players=%d, n_matches=%d, "
            "last max_gamma_diff=%.2e, last theta_diff=%.2e); ratings may be unstable.",
            max_iter, n, len(matches), max_diff, abs(theta - theta_old),
        )

    log_gamma = [math.log(g + 1e-15) for g in gamma]
    log_mean = sum(log_gamma) / n
    return {
        p: (log_gamma[idx[p]] - log_mean) * 400 / math.log(10) + 1500
        for p in players
    }


def _win_loss_tie(matches: list[ArenaMatch]) -> dict[str, dict[str, int]]:
    """Per-model decisive wins/losses/ties derived from debiased scores."""
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0})
    for match in matches:
        s = match.debiased_score
        a, b = match.model_a, match.model_b
        if s > 0.5:
            stats[a]["win"] += 1
            stats[b]["loss"] += 1
        elif s < 0.5:
            stats[b]["win"] += 1
            stats[a]["loss"] += 1
        else:
            stats[a]["tie"] += 1
            stats[b]["tie"] += 1
    return dict(stats)


def build_leaderboard(
    *,
    config: str,
    domain: DomainConfig,
    judge_profile: str,
    matches: list[ArenaMatch],
) -> Leaderboard:
    """Assemble an arena leaderboard from per-case matches."""
    ratings = compute_btd(matches)
    stats = _win_loss_tie(matches)
    consistency = [m.consistent for m in matches]
    consistency_rate = sum(consistency) / len(consistency) if consistency else None

    ranked = sorted(ratings.items(), key=lambda x: -x[1])
    entries = [
        LeaderboardEntry(
            breakdown={k: float(v) for k, v in stats.get(model, {}).items()},
            model=model,
            n_observations=stats.get(model, {}).get("win", 0)
            + stats.get(model, {}).get("loss", 0)
            + stats.get(model, {}).get("tie", 0),
            rank=rank,
            rating=rating,
        )
        for rank, (model, rating) in enumerate(ranked, start=1)
    ]

    metadata = LeaderboardMetadata(
        config=config,
        domain=domain.name,
        judge=judge_profile,
        method="arena",
        n_models=len(entries),
        n_observations=len(matches),
        position_consistency_rate=consistency_rate,
    )
    return Leaderboard(metadata=metadata, rankings=entries)


def filter_matches_by_pool(matches: list[ArenaMatch], pool: Pool) -> list[ArenaMatch]:
    """Keep only matches whose both sides belong to ``pool`` (reference is always retained).

    For ``pool='full'`` returns the input list unchanged (no copy) — the filter would
    accept every entry, so allocating a duplicate list wastes ~10ms on 40k matches.
    Callers must therefore treat the result as read-only.
    """
    if pool == "full":
        return matches
    return [
        m for m in matches
        if keep_in_pool(m.model_a, pool) and keep_in_pool(m.model_b, pool)
    ]


def build_pool_leaderboard(
    *,
    config: str,
    domain: DomainConfig,
    judge_profile: str,
    matches: list[ArenaMatch],
    pool: Pool,
) -> Leaderboard:
    """Build a BTD leaderboard restricted to ``pool``; metadata.pool is set accordingly."""
    sub = filter_matches_by_pool(matches, pool)
    leaderboard = build_leaderboard(
        config=config, domain=domain, judge_profile=judge_profile, matches=sub,
    )
    leaderboard.metadata.pool = pool
    return leaderboard


def encode_match(match: ArenaMatch) -> str:
    """Serialize one ArenaMatch to a JSON line."""
    return json.dumps(match.model_dump(mode="json"), ensure_ascii=False)


def encode_leaderboard(leaderboard: Leaderboard) -> str:
    """Serialize a Leaderboard to a single JSON string (indented)."""
    return json.dumps(leaderboard.model_dump(mode="json"), ensure_ascii=False, indent=2)
