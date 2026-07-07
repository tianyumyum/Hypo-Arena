"""Orchestrator CLI: run the full benchmark pipeline (construction → gen → arena+score) end-to-end.

The orchestrator hot-pickups across stages, retries failed items indefinitely with jittered
exponential backoff, and auto-rebuilds leaderboards as new data arrives.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import resource
import sys
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import ALL_DOMAINS, configure_runtime
from orchestrator import OrchestratorConfig, run_orchestrator


def _raise_fd_limit(target: int = 16384) -> None:
    """Bump RLIMIT_NOFILE up to `target` (or hard cap, whichever is lower).

    macOS launchd hands out a soft limit of 256 fds, which is far below what 360+
    async workers × 4 platform HTTPS clients consume. Without raising it, every new
    socket eventually fails with OSError(24) Too many open files, surfacing as
    APIConnectionError throughout the run.
    """
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    desired_hard = hard if hard != resource.RLIM_INFINITY else target
    new_soft = min(target, desired_hard)
    if new_soft <= soft:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
    except (ValueError, OSError) as exc:
        logging.getLogger("hypo.orchestrate").warning(
            "could not raise RLIMIT_NOFILE from %d to %d: %s", soft, new_soft, exc,
        )

DEFAULT_GEN_PROFILES = (
    "claude-sonnet-4.6-high",
    "claude-opus-4.6-high",
    "deepseek-v4-flash-high",
    "deepseek-v4-pro-high",
    "gemini-3-flash-high",
    "gemini-3.1-pro-high",
    "glm-5-thinking",
    "glm-5.1-thinking",
    "gpt-5.4-mini-high",
    "gpt-5.4-high",
    "kimi-k2.5-thinking",
    "kimi-k2.6-thinking",
    "minimax-m2.5-thinking",
    "minimax-m2.7-thinking",
    "qwen-3.6-max-thinking",
    "qwen-3.7-max-thinking",
)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _parse_domains(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(ALL_DOMAINS)
    out = _csv(value)
    bad = [d for d in out if d not in ALL_DOMAINS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown domains: {bad}; valid: {sorted(ALL_DOMAINS)}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HypoArena pipeline orchestrator")

    parser.add_argument("--domains", type=_parse_domains, default="all",
                        help="comma-list or 'all' (default: all 6)")
    parser.add_argument("--construction-profile", default="gpt-5.4-high")
    parser.add_argument("--gen-profiles", type=_csv, default=",".join(DEFAULT_GEN_PROFILES),
                        help="comma-list of generation profiles")
    parser.add_argument("--gen-modes", type=_csv, default="baseline,agent",
                        help="subset of {baseline, agent}")
    parser.add_argument("--judges", type=_csv, default="mimo-v2-pro",
                        dest="judge_profiles",
                        help="comma-list of judge profiles; each gets its own parallel arena+score files")
    parser.add_argument("--with-recall", action="store_true", default=True,
                        help="reference-anchored recall in score (multi-hypothesis only); on by default")
    parser.add_argument("--no-recall", action="store_false", dest="with_recall")

    # concurrency
    parser.add_argument("--construction-concurrency", type=int, default=4,
                        help="workers per domain (× n_domains total)")
    parser.add_argument("--generation-concurrency", type=int, default=1,
                        help="workers per (domain × mode × profile) triple")
    parser.add_argument("--arena-concurrency", type=int, default=16,
                        help="workers per domain")
    parser.add_argument("--score-concurrency", type=int, default=8,
                        help="workers per domain")

    # robustness
    parser.add_argument("--max-rounds", type=int, default=4)
    parser.add_argument("--task-timeout-construction", type=float, default=900.0)
    parser.add_argument("--task-timeout-generation", type=float, default=900.0)
    parser.add_argument("--task-timeout-arena", type=float, default=300.0)
    parser.add_argument("--task-timeout-score", type=float, default=300.0)

    # timing
    parser.add_argument("--poll-interval", type=float, default=10.0)
    parser.add_argument("--leaderboard-interval", type=float, default=60.0)

    # modes
    parser.add_argument("--dry-run", action="store_true",
                        help="scan + report pending counts, do not execute")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    cfg = OrchestratorConfig(
        arena_concurrency=args.arena_concurrency,
        construction_concurrency=args.construction_concurrency,
        construction_profile=args.construction_profile,
        domains=args.domains,
        dry_run=args.dry_run,
        gen_modes=args.gen_modes,
        gen_profiles=tuple(args.gen_profiles),
        generation_concurrency=args.generation_concurrency,
        judge_profiles=tuple(args.judge_profiles),
        leaderboard_interval=args.leaderboard_interval,
        max_rounds=args.max_rounds,
        poll_interval=args.poll_interval,
        score_concurrency=args.score_concurrency,
        task_timeout_arena=args.task_timeout_arena,
        task_timeout_construction=args.task_timeout_construction,
        task_timeout_generation=args.task_timeout_generation,
        task_timeout_score=args.task_timeout_score,
        with_recall=args.with_recall,
    )

    _raise_fd_limit()
    configure_runtime()
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    logging.getLogger("hypo.orchestrate").info("RLIMIT_NOFILE soft=%d hard=%s", soft, hard)
    try:
        asyncio.run(run_orchestrator(cfg))
    except KeyboardInterrupt:
        logging.getLogger("hypo.orchestrate").warning("interrupted (hard exit)")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
