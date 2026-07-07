"""Stage supervisors: per-domain construction / generation / arena / score + leaderboard rebuilder.

Each stage type runs **one Stage instance per domain**, so concurrency is dedicated:
e.g. `--construction-concurrency 4` → 4 workers per domain × 6 domains = 24 truly parallel.
Per-domain stages are decoupled: generation for biomedical doesn't wait for safety construction.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from basics import (
    BenchmarkCase,
    Submission,
    get_domain,
)
from basics.io import (
    append_arena_match,
    append_case,
    append_score_record,
    append_submission,
    load_all_submissions,
    load_arena_matches,
    load_cases,
    load_score_records,
    load_sources,
)
from basics.paths import (
    arena_matches_path,
    arena_pool_leaderboard_path,
    cases_path,
    score_pool_leaderboard_path,
    score_records_path,
    submission_path,
)
from construction import construct_case
from evaluation import (
    build_arena_pool_leaderboard,
    build_score_pool_leaderboard,
    judge_pair,
    reference_submission,
    score_submission,
)
from generation import generate_submission

from evaluation.markdown import render_leaderboard_md

from .atomic import write_json_atomic, write_text_atomic
from .keys import raw_arena_pair_keys, raw_existing_ids, raw_score_keys
from .stage import Stage, StageConfig, StageItem, StageObserver

logger = logging.getLogger("hypo.orchestrator.supervisors")


# ---- config ----

@dataclass
class OrchestratorConfig:
    """Top-level orchestrator configuration; populated from CLI flags."""
    arena_concurrency: int = 16             # per-domain workers (× n_domains total)
    construction_concurrency: int = 4       # per-domain workers
    construction_profile: str = "gpt-5.4-high"
    domains: tuple[str, ...] = ()
    dry_run: bool = False
    gen_modes: tuple[str, ...] = ("baseline", "agent")
    gen_profiles: tuple[str, ...] = ()
    generation_concurrency: int = 1         # per-(domain×mode×profile) workers
    judge_profiles: tuple[str, ...] = ()    # parallel judges; each produces its own arena+score files
    leaderboard_interval: float = 60.0
    max_rounds: int = 4
    poll_interval: float = 10.0
    score_concurrency: int = 8
    task_timeout_arena: float = 300.0
    task_timeout_construction: float = 900.0
    task_timeout_generation: float = 900.0
    task_timeout_score: float = 300.0
    with_recall: bool = True


# ---- construction (per domain) ----

def build_construction_stage_for_domain(
    *,
    cfg: OrchestratorConfig,
    observer: StageObserver,
    domain_name: str,
) -> Stage:
    """One construction Stage dedicated to a single domain; no upstream."""
    domain_cfg = get_domain(domain_name)

    def scan():
        path = cases_path(domain_name, cfg.construction_profile)
        done = raw_existing_ids(path)
        for src in load_sources(domain_name):
            if src.id in done:
                continue
            yield StageItem(
                domain=domain_name,
                key=f"construction:{domain_name}:{src.id}",
                short_label=src.id,
                payload={"record": src},
            )

    async def work(item: StageItem) -> None:
        case = await construct_case(
            domain=domain_cfg,
            forge_profile=cfg.construction_profile,
            max_rounds=cfg.max_rounds,
            record=item.payload["record"],
        )
        append_case(item.domain, cfg.construction_profile, case)

    upstream_done = asyncio.Event()
    upstream_done.set()                                 # construction has no upstream
    stage_cfg = StageConfig(
        name="construction",
        concurrency=cfg.construction_concurrency,
        task_timeout=cfg.task_timeout_construction,
        poll_interval=cfg.poll_interval,
        queue_maxsize=0,                                  # unbounded; scan_fn materializes upfront anyway
    )
    return Stage(
        cfg=stage_cfg,
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream_done,
        work_fn=work,
    )


# ---- generation (per domain × mode × profile) ----

def build_generation_stage_for_triple(
    *,
    cfg: OrchestratorConfig,
    observer: StageObserver,
    domain_name: str,
    mode: str,
    profile: str,
    upstream_done: asyncio.Event,
) -> Stage:
    """One generation Stage per (domain, mode, profile); upstream = that domain's construction.done."""
    domain_cfg = get_domain(domain_name)

    def scan():
        cases = load_cases(domain_name, cfg.construction_profile)
        if not cases:
            return
        path = submission_path(domain_name, mode, cfg.construction_profile, profile)
        done = raw_existing_ids(path)
        for case in cases:
            if case.id in done:
                continue
            yield StageItem(
                domain=domain_name,
                key=f"generation:{domain_name}:{case.id}:{mode}:{profile}",
                short_label=f"{case.id}×{profile}[{mode}]",
                payload={"case": case, "mode": mode, "profile": profile},
            )

    async def work(item: StageItem) -> None:
        payload = item.payload
        sub = await generate_submission(
            case=payload["case"],
            domain=domain_cfg,
            mode=payload["mode"],
            profile_name=payload["profile"],
        )
        append_submission(item.domain, payload["mode"], cfg.construction_profile, payload["profile"], sub)

    stage_cfg = StageConfig(
        name="generation",
        concurrency=cfg.generation_concurrency,
        task_timeout=cfg.task_timeout_generation,
        poll_interval=cfg.poll_interval,
        queue_maxsize=0,
    )
    return Stage(
        cfg=stage_cfg,
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream_done,
        work_fn=work,
    )


# ---- shared helper ----

def _collect_available(
    cases: list[BenchmarkCase],
    *,
    domain_name: str,
    cfg: OrchestratorConfig,
) -> dict[str, dict[str, Submission]]:
    """Per case_id → {label: Submission}. Label is 'reference' or '{mode}:{profile}'."""
    by_case: dict[str, dict[str, Submission]] = defaultdict(dict)
    for case in cases:
        by_case[case.id]["reference"] = reference_submission(case)
    for mode in cfg.gen_modes:
        for profile, subs in load_all_submissions(domain_name, mode, cfg.construction_profile).items():
            label = f"{mode}:{profile}"
            for sub in subs:
                if sub.id in by_case:
                    by_case[sub.id][label] = sub
    return by_case


# ---- arena (per domain) ----

def build_arena_stage_for_domain(
    *,
    cfg: OrchestratorConfig,
    observer: StageObserver,
    domain_name: str,
    judge_profile: str,
    upstream_done: asyncio.Event,
) -> Stage:
    """One arena Stage per (domain, judge); pairs fan out incrementally as submissions arrive."""
    domain_cfg = get_domain(domain_name)

    def scan():
        cases = load_cases(domain_name, cfg.construction_profile)
        if not cases:
            return
        by_case = _collect_available(cases, domain_name=domain_name, cfg=cfg)
        done_pairs = raw_arena_pair_keys(
            arena_matches_path(domain_name, cfg.construction_profile, judge_profile)
        )
        case_by_id = {c.id: c for c in cases}
        for case_id, avail in by_case.items():
            if len(avail) < 2:
                continue
            labels_sorted = sorted(avail)
            for a, b in itertools.combinations(labels_sorted, 2):
                if (case_id, frozenset([a, b])) in done_pairs:
                    continue
                yield StageItem(
                    domain=domain_name,
                    key=f"arena:{judge_profile}:{domain_name}:{case_id}:{a}|{b}",
                    short_label=f"{case_id} {a} vs {b}",
                    payload={
                        "case": case_by_id[case_id],
                        "a": a, "b": b,
                        "sub_a": avail[a], "sub_b": avail[b],
                    },
                )

    async def work(item: StageItem) -> None:
        payload = item.payload
        match = await judge_pair(
            case=payload["case"],
            domain=domain_cfg,
            judge_profile=judge_profile,
            model_a=payload["a"],
            model_b=payload["b"],
            submission_a=payload["sub_a"],
            submission_b=payload["sub_b"],
        )
        append_arena_match(item.domain, cfg.construction_profile, judge_profile, match)

    stage_cfg = StageConfig(
        name=f"arena.{judge_profile}",
        concurrency=cfg.arena_concurrency,
        task_timeout=cfg.task_timeout_arena,
        poll_interval=cfg.poll_interval,
        queue_maxsize=0,
    )
    return Stage(
        cfg=stage_cfg,
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream_done,
        work_fn=work,
    )


# ---- score (per domain) ----

def build_score_stage_for_domain(
    *,
    cfg: OrchestratorConfig,
    observer: StageObserver,
    domain_name: str,
    judge_profile: str,
    upstream_done: asyncio.Event,
) -> Stage:
    """One score Stage per (domain, judge); fans out per (case, model_label)."""
    domain_cfg = get_domain(domain_name)
    use_recall = cfg.with_recall and domain_cfg.multi_hypothesis

    def scan():
        cases = load_cases(domain_name, cfg.construction_profile)
        if not cases:
            return
        by_case = _collect_available(cases, domain_name=domain_name, cfg=cfg)
        done = raw_score_keys(
            score_records_path(domain_name, cfg.construction_profile, judge_profile)
        )
        case_by_id = {c.id: c for c in cases}
        for case_id, avail in by_case.items():
            for label, sub in avail.items():
                if (case_id, label) in done:
                    continue
                yield StageItem(
                    domain=domain_name,
                    key=f"score:{judge_profile}:{domain_name}:{case_id}:{label}",
                    short_label=f"{case_id} × {label}",
                    payload={"case": case_by_id[case_id], "label": label, "sub": sub},
                )

    async def work(item: StageItem) -> None:
        payload = item.payload
        record = await score_submission(
            case=payload["case"],
            domain=domain_cfg,
            judge_profile=judge_profile,
            model=payload["label"],
            submission=payload["sub"],
            with_recall=use_recall,
        )
        append_score_record(item.domain, cfg.construction_profile, judge_profile, record)

    stage_cfg = StageConfig(
        name=f"score.{judge_profile}",
        concurrency=cfg.score_concurrency,
        task_timeout=cfg.task_timeout_score,
        poll_interval=cfg.poll_interval,
        queue_maxsize=0,
    )
    return Stage(
        cfg=stage_cfg,
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream_done,
        work_fn=work,
    )


# ---- leaderboard rebuilder (single, sweeps all (domain × judge × pool) combinations) ----

POOLS: tuple[str, ...] = ("baseline", "agent", "full")


@dataclass
class _LeaderboardCache:
    # Cache key includes pool so rebuilds for one pool don't suppress siblings.
    arena_mtime: dict[tuple[str, str], float] = field(default_factory=dict)   # keyed by (domain, judge)
    score_mtime: dict[tuple[str, str], float] = field(default_factory=dict)


def _rebuild_arena_leaderboard(domain_name: str, judge_profile: str, cfg: OrchestratorConfig) -> None:
    """Rebuild all 3 pool leaderboards (baseline / agent / full) for one (domain, judge)."""
    matches = load_arena_matches(domain_name, cfg.construction_profile, judge_profile)
    if not matches:
        return
    domain_cfg = get_domain(domain_name)
    for pool in POOLS:
        lb = build_arena_pool_leaderboard(
            config=cfg.construction_profile,
            domain=domain_cfg,
            judge_profile=judge_profile,
            matches=matches,
            pool=pool,
        )
        write_json_atomic(
            arena_pool_leaderboard_path(domain_name, cfg.construction_profile, judge_profile, pool),
            lb,
        )
        write_text_atomic(
            arena_pool_leaderboard_path(
                domain_name, cfg.construction_profile, judge_profile, pool, suffix="md",
            ),
            render_leaderboard_md(lb),
        )


def _rebuild_score_leaderboard(domain_name: str, judge_profile: str, cfg: OrchestratorConfig) -> None:
    """Rebuild all 3 pool score leaderboards for one (domain, judge)."""
    records = load_score_records(domain_name, cfg.construction_profile, judge_profile)
    if not records:
        return
    domain_cfg = get_domain(domain_name)
    for pool in POOLS:
        lb = build_score_pool_leaderboard(
            config=cfg.construction_profile,
            domain=domain_cfg,
            judge_profile=judge_profile,
            records=records,
            pool=pool,
        )
        write_json_atomic(
            score_pool_leaderboard_path(domain_name, cfg.construction_profile, judge_profile, pool),
            lb,
        )
        write_text_atomic(
            score_pool_leaderboard_path(
                domain_name, cfg.construction_profile, judge_profile, pool, suffix="md",
            ),
            render_leaderboard_md(lb),
        )


async def leaderboard_loop(
    *,
    cfg: OrchestratorConfig,
    arena_done: asyncio.Event,
    score_done: asyncio.Event,
) -> None:
    """Periodic rebuild per (domain, judge, method); only when the source mtime changes."""
    cache = _LeaderboardCache()
    while True:
        for domain_name in cfg.domains:
            for judge in cfg.judge_profiles:
                key = (domain_name, judge)
                try:
                    arena_src = arena_matches_path(domain_name, cfg.construction_profile, judge)
                    if arena_src.exists():
                        mtime = arena_src.stat().st_mtime
                        if cache.arena_mtime.get(key) != mtime:
                            _rebuild_arena_leaderboard(domain_name, judge, cfg)
                            cache.arena_mtime[key] = mtime
                except Exception:
                    logger.exception("arena leaderboard rebuild failed domain=%s judge=%s",
                                     domain_name, judge)
                try:
                    score_src = score_records_path(domain_name, cfg.construction_profile, judge)
                    if score_src.exists():
                        mtime = score_src.stat().st_mtime
                        if cache.score_mtime.get(key) != mtime:
                            _rebuild_score_leaderboard(domain_name, judge, cfg)
                            cache.score_mtime[key] = mtime
                except Exception:
                    logger.exception("score leaderboard rebuild failed domain=%s judge=%s",
                                     domain_name, judge)

        if arena_done.is_set() and score_done.is_set():
            for domain_name in cfg.domains:
                for judge in cfg.judge_profiles:
                    try:
                        _rebuild_arena_leaderboard(domain_name, judge, cfg)
                        _rebuild_score_leaderboard(domain_name, judge, cfg)
                    except Exception:
                        logger.exception("final leaderboard rebuild failed domain=%s judge=%s",
                                         domain_name, judge)
            return
        await asyncio.sleep(cfg.leaderboard_interval)
