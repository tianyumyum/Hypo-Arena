"""Orchestrator: per-domain pipeline (construction → generation → arena + score)."""

from __future__ import annotations

import asyncio
import logging
import signal
import time
from pathlib import Path

from basics.paths import ARTIFACTS_ROOT
from basics.schema import Leaderboard
from evaluation.markdown import render_leaderboard_md

from .atomic import write_json_atomic, write_text_atomic
from .keys import invalidate_cache, raw_arena_pair_keys, raw_existing_ids, raw_score_keys
from .stage import Stage, StageConfig, StageItem, StageObserver
from .supervisors import (
    OrchestratorConfig,
    build_arena_stage_for_domain,
    build_construction_stage_for_domain,
    build_generation_stage_for_triple,
    build_score_stage_for_domain,
    leaderboard_loop,
)

logger = logging.getLogger("hypo.orchestrator")

STAGES: tuple[str, ...] = ("construction", "generation", "arena", "score")


_LEGACY_LEADERBOARD_SUFFIXES: tuple[str, ...] = (
    ".arena.leaderboard.json",
    ".score.leaderboard.json",
)


def _is_legacy_leaderboard(path: Path) -> bool:
    """True when the path is a pre-pool-migration leaderboard (no `.{pool}.` segment)."""
    return any(path.name.endswith(s) for s in _LEGACY_LEADERBOARD_SUFFIXES)


def _backfill_leaderboard_markdown() -> tuple[int, int]:
    """Regenerate `.md` companions for every existing pool-aware `*.leaderboard.json` whose
    markdown twin is missing or older than the source JSON.

    Pre-Stage-E legacy leaderboard files (`*.arena.leaderboard.json` / `*.score.leaderboard.json`,
    no `.baseline.` / `.agent.` / `.full.` segment) are intentionally skipped — they are no longer
    produced and their content is stale; running ``scripts/cleanup_legacy_leaderboards.py`` removes
    them physically.

    Returns (rebuilt_count, skipped_count). Called once at orchestrator startup so
    leaderboards built before the markdown emitter existed get caught up without
    waiting for the next live rebuild.
    """
    rebuilt = 0
    skipped = 0
    for json_path in ARTIFACTS_ROOT.glob("*/results/*.leaderboard.json"):
        if _is_legacy_leaderboard(json_path):
            skipped += 1
            continue
        md_path = json_path.with_suffix(".md")
        if md_path.exists() and md_path.stat().st_mtime >= json_path.stat().st_mtime:
            skipped += 1
            continue
        try:
            lb = Leaderboard.model_validate_json(json_path.read_text(encoding="utf-8"))
        except Exception:
            logger.exception("backfill: cannot parse %s", json_path)
            continue
        write_text_atomic(md_path, render_leaderboard_md(lb))
        rebuilt += 1
    return rebuilt, skipped


def _initial_pending(cfg: OrchestratorConfig) -> dict[str, dict[str, int]]:
    """Count pending items per (stage, domain) before starting. For --dry-run."""
    import itertools

    from basics.io import load_cases, load_sources
    from basics.paths import (
        arena_matches_path,
        cases_path,
        score_records_path,
        submission_path,
    )

    from .supervisors import _collect_available

    out: dict[str, dict[str, int]] = {s: {} for s in STAGES}
    for d in cfg.domains:
        c_done = raw_existing_ids(cases_path(d, cfg.construction_profile))
        sources = load_sources(d)
        out["construction"][d] = sum(1 for s in sources if s.id not in c_done)

        cases = load_cases(d, cfg.construction_profile)
        case_ids = {c.id for c in cases}

        gen_pending = 0
        for mode in cfg.gen_modes:
            for profile in cfg.gen_profiles:
                done = raw_existing_ids(submission_path(d, mode, cfg.construction_profile, profile))
                gen_pending += sum(1 for cid in case_ids if cid not in done)
        out["generation"][d] = gen_pending

        by_case = _collect_available(cases, domain_name=d, cfg=cfg) if cases else {}
        arena_pending = 0
        score_pending = 0
        for judge in cfg.judge_profiles:
            done_pairs = raw_arena_pair_keys(arena_matches_path(d, cfg.construction_profile, judge))
            done_scores = raw_score_keys(score_records_path(d, cfg.construction_profile, judge))
            for case_id, avail in by_case.items():
                if len(avail) >= 2:
                    for a, b in itertools.combinations(sorted(avail), 2):
                        if (case_id, frozenset([a, b])) not in done_pairs:
                            arena_pending += 1
                for label in avail:
                    if (case_id, label) not in done_scores:
                        score_pending += 1
        out["arena"][d] = arena_pending
        out["score"][d] = score_pending
    return out


async def _set_when_all_done(events: list[asyncio.Event], target: asyncio.Event) -> None:
    """Helper: fire `target` once every event in `events` has been set."""
    for e in events:
        await e.wait()
    target.set()


async def run_orchestrator(cfg: OrchestratorConfig) -> None:
    """Top-level entrypoint: per-domain stages × 4 stage types + leaderboard rebuilder.

    Failure handling: every item retries indefinitely with jittered exponential backoff
    (capped by StageConfig.backoff_max, ~32 min). There is no quarantine — a permanently
    failing item keeps the orchestrator running until the operator Ctrl+Cs.
    """
    if cfg.dry_run:
        _print_dry_run_report(cfg)
        return

    rebuilt, skipped = _backfill_leaderboard_markdown()
    if rebuilt or skipped:
        logger.info("leaderboard md backfill: rebuilt=%d already_fresh=%d", rebuilt, skipped)

    started = time.time()
    observer = StageObserver()                                      # no-op; metrics come from artifacts/

    # Build per-domain construction stages.
    construction_by_domain: dict[str, Stage] = {
        d: build_construction_stage_for_domain(
            cfg=cfg, observer=observer, domain_name=d,
        )
        for d in cfg.domains
    }

    # Build per-(domain, mode, profile) generation stages; upstream = that domain's construction.done.
    generation_by_triple: dict[tuple[str, str, str], Stage] = {}
    for d in cfg.domains:
        for mode in cfg.gen_modes:
            for profile in cfg.gen_profiles:
                generation_by_triple[(d, mode, profile)] = build_generation_stage_for_triple(
                    cfg=cfg, observer=observer,
                    domain_name=d, mode=mode, profile=profile,
                    upstream_done=construction_by_domain[d].done,
                )

    # Per-domain "all generation triples done" event, used by arena/score upstream.
    gen_all_done_by_domain: dict[str, asyncio.Event] = {d: asyncio.Event() for d in cfg.domains}
    for d in cfg.domains:
        triples_for_d = [
            generation_by_triple[(d, mode, profile)]
            for mode in cfg.gen_modes
            for profile in cfg.gen_profiles
        ]
        asyncio.create_task(
            _set_when_all_done([s.done for s in triples_for_d], gen_all_done_by_domain[d]),
            name=f"gen_done_combine.{d}",
        )

    arena_by_key: dict[tuple[str, str], Stage] = {
        (d, j): build_arena_stage_for_domain(
            cfg=cfg, observer=observer, domain_name=d, judge_profile=j,
            upstream_done=gen_all_done_by_domain[d],
        )
        for d in cfg.domains
        for j in cfg.judge_profiles
    }
    score_by_key: dict[tuple[str, str], Stage] = {
        (d, j): build_score_stage_for_domain(
            cfg=cfg, observer=observer, domain_name=d, judge_profile=j,
            upstream_done=gen_all_done_by_domain[d],
        )
        for d in cfg.domains
        for j in cfg.judge_profiles
    }

    shutdown = asyncio.Event()
    _install_signal_handlers(shutdown)

    # Spawn one task per Stage instance.
    supervisor_tasks: list[asyncio.Task] = []
    for d, stage in construction_by_domain.items():
        supervisor_tasks.append(asyncio.create_task(stage.run(), name=f"sup.construction.{d}"))
    for (d, mode, profile), stage in generation_by_triple.items():
        supervisor_tasks.append(
            asyncio.create_task(stage.run(), name=f"sup.generation.{d}.{mode}.{profile}"),
        )
    for (d, j), stage in arena_by_key.items():
        supervisor_tasks.append(asyncio.create_task(stage.run(), name=f"sup.arena.{d}.{j}"))
    for (d, j), stage in score_by_key.items():
        supervisor_tasks.append(asyncio.create_task(stage.run(), name=f"sup.score.{d}.{j}"))

    # Combined per-pipeline-stage done events for the leaderboard loop.
    arena_all_done = asyncio.Event()
    score_all_done = asyncio.Event()
    asyncio.create_task(_set_when_all_done([s.done for s in arena_by_key.values()], arena_all_done))
    asyncio.create_task(_set_when_all_done([s.done for s in score_by_key.values()], score_all_done))

    leaderboard_task = asyncio.create_task(
        leaderboard_loop(cfg=cfg, arena_done=arena_all_done, score_done=score_all_done),
        name="sup.leaderboard",
    )

    shutdown_watcher = asyncio.create_task(
        _watch_shutdown(shutdown, supervisor_tasks + [leaderboard_task]),
    )

    try:
        await asyncio.gather(*supervisor_tasks)
        await leaderboard_task
    finally:
        shutdown_watcher.cancel()
        invalidate_cache()

    _summarize(started)


async def _watch_shutdown(shutdown: asyncio.Event, tasks: list[asyncio.Task]) -> None:
    """On SIGINT: let tasks finish in-flight work, then cancel after a grace period."""
    await shutdown.wait()
    logger.warning("shutdown requested; allowing 5s grace for in-flight work before cancel")
    await asyncio.sleep(5.0)
    for t in tasks:
        if not t.done():
            t.cancel()


def _install_signal_handlers(shutdown: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()

    def _handler():
        if shutdown.is_set():
            logger.error("second interrupt; exiting hard")
            raise KeyboardInterrupt
        logger.warning("interrupt received; setting shutdown event")
        shutdown.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handler)
        except (NotImplementedError, RuntimeError):
            pass


def _summarize(started: float) -> None:
    elapsed = time.time() - started
    line = f"Orchestrator finished in {elapsed / 60:.1f} min."
    logger.info(line)
    print(line)


def _print_dry_run_report(cfg: OrchestratorConfig) -> None:
    pending = _initial_pending(cfg)
    n_d = len(cfg.domains)
    n_modes = len(cfg.gen_modes)
    n_profiles = len(cfg.gen_profiles)
    n_judges = len(cfg.judge_profiles)
    n_g_triples = n_d * n_modes * n_profiles
    print(f"Dry run: {n_d} domain(s), {n_profiles} gen profiles, {n_modes} modes, {n_judges} judges")
    print(f"  construction = {cfg.construction_concurrency} workers/domain × {n_d} = "
          f"{cfg.construction_concurrency * n_d} total")
    print(f"  generation   = {cfg.generation_concurrency} workers/(domain×mode×profile) × {n_g_triples} = "
          f"{cfg.generation_concurrency * n_g_triples} total")
    print(f"  arena        = {cfg.arena_concurrency} workers/(domain×judge) × {n_d * n_judges} = "
          f"{cfg.arena_concurrency * n_d * n_judges} total")
    print(f"  score        = {cfg.score_concurrency} workers/(domain×judge) × {n_d * n_judges} = "
          f"{cfg.score_concurrency * n_d * n_judges} total")
    grand = (
        cfg.construction_concurrency * n_d
        + cfg.generation_concurrency * n_g_triples
        + cfg.arena_concurrency * n_d * n_judges
        + cfg.score_concurrency * n_d * n_judges
    )
    print(f"  GRAND TOTAL  = {grand} workers")
    print()
    for stage in STAGES:
        totals = pending[stage]
        grand_p = sum(totals.values())
        suffix = f" (× {n_judges} judges)" if stage in ("arena", "score") else ""
        print(f"  {stage:12s} pending total={grand_p}{suffix}")
        for d, n in totals.items():
            if n:
                print(f"     {d:25s} {n}")


__all__ = [
    "OrchestratorConfig",
    "STAGES",
    "Stage",
    "StageConfig",
    "StageItem",
    "StageObserver",
    "build_arena_stage_for_domain",
    "build_construction_stage_for_domain",
    "build_generation_stage_for_triple",
    "build_score_stage_for_domain",
    "leaderboard_loop",
    "run_orchestrator",
    "write_json_atomic",
]
