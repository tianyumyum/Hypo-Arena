"""Rebuild pool-filtered arena + score leaderboards from raw matches.jsonl / score.jsonl.

Walks every `(domain, judge)` pair under `artifacts/*/results/`, reads the raw arena
matches and score records once, then writes 3 arena leaderboards + 3 score leaderboards
(`baseline`, `agent`, `full`) as `*.arena.{pool}.leaderboard.{json,md}` /
`*.score.{pool}.leaderboard.{json,md}`. Does NOT touch the raw jsonl files, and does
NOT overwrite the pool-agnostic legacy `*.arena.leaderboard.*` / `*.score.leaderboard.*`
files.

Idempotent — safe to run repeatedly. Intended as the offline bridge before the
orchestrator's `_rebuild_*_leaderboard` hot path is updated (Stage E).

Usage:
  uv run python scripts/rebuild_pool_leaderboards.py
  uv run python scripts/rebuild_pool_leaderboards.py --domains biomedical_science,it_operations
  uv run python scripts/rebuild_pool_leaderboards.py --judges mimo-v2-pro
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import (
    ALL_DOMAINS,
    ARTIFACTS_ROOT,
    arena_pool_leaderboard_path,
    configure_runtime,
    get_domain,
    score_pool_leaderboard_path,
)
from basics.io import load_arena_matches, load_score_records
from basics.models import CONSTRUCTION_REGISTRY
from evaluation import build_arena_pool_leaderboard, build_score_pool_leaderboard
from evaluation.markdown import render_leaderboard_md
from orchestrator.atomic import write_json_atomic, write_text_atomic

logger = logging.getLogger("hypo.rebuild_pool")

POOLS: tuple[str, ...] = ("baseline", "agent", "full")

_ARENA_SUFFIX = ".arena.matches.jsonl"
_SCORE_SUFFIX = ".score.jsonl"
# Some judge profiles also contain dots (e.g. "seed-2.0-pro") and construction
# profiles already do (e.g. "gpt-5.4"), so file names cannot be parsed by a
# naive dot-split. We instead find the longest known construction-profile
# prefix via CONSTRUCTION_REGISTRY.
_KNOWN_CONFIGS: tuple[str, ...] = tuple(
    sorted(CONSTRUCTION_REGISTRY, key=len, reverse=True)
)


def _split_config_judge(prefix: str) -> tuple[str, str] | None:
    """Split `<config>.<judge>` against the known construction-profile registry."""
    for config in _KNOWN_CONFIGS:
        if prefix == config:
            return None
        if prefix.startswith(config + "."):
            return config, prefix[len(config) + 1:]
    return None


def _discover_runs(
    domains: tuple[str, ...], *, suffix: str,
) -> set[tuple[str, str, str]]:
    """Find every (domain, config, judge) for files ending in ``suffix``."""
    found: set[tuple[str, str, str]] = set()
    for domain in domains:
        results = ARTIFACTS_ROOT / domain / "results"
        if not results.exists():
            continue
        for path in results.iterdir():
            if not path.name.endswith(suffix):
                continue
            prefix = path.name[: -len(suffix)]
            split = _split_config_judge(prefix)
            if split is None:
                logger.debug("skip unparseable file: %s", path.name)
                continue
            found.add((domain, split[0], split[1]))
    return found


def _discover_arena_runs(domains: tuple[str, ...]) -> set[tuple[str, str, str]]:
    return _discover_runs(domains, suffix=_ARENA_SUFFIX)


def _discover_score_runs(domains: tuple[str, ...]) -> set[tuple[str, str, str]]:
    return _discover_runs(domains, suffix=_SCORE_SUFFIX)


def _filter_runs(
    runs: set[tuple[str, str, str]],
    *,
    configs: tuple[str, ...] | None,
    judges: tuple[str, ...] | None,
) -> list[tuple[str, str, str]]:
    keep = list(runs)
    if configs is not None:
        keep = [r for r in keep if r[1] in configs]
    if judges is not None:
        keep = [r for r in keep if r[2] in judges]
    return sorted(keep)


def rebuild_arena(*, domain: str, config: str, judge: str) -> None:
    matches = load_arena_matches(domain, config, judge)
    if not matches:
        logger.info("arena.skip domain=%s judge=%s (no matches)", domain, judge)
        return
    domain_cfg = get_domain(domain)
    for pool in POOLS:
        lb = build_arena_pool_leaderboard(
            config=config, domain=domain_cfg, judge_profile=judge,
            matches=matches, pool=pool,
        )
        write_json_atomic(arena_pool_leaderboard_path(domain, config, judge, pool), lb)
        write_text_atomic(
            arena_pool_leaderboard_path(domain, config, judge, pool, suffix="md"),
            render_leaderboard_md(lb),
        )
        logger.info(
            "arena.ok domain=%s judge=%s pool=%s n_models=%d n_obs=%d",
            domain, judge, pool, lb.metadata.n_models, lb.metadata.n_observations,
        )


def rebuild_score(*, domain: str, config: str, judge: str) -> None:
    records = load_score_records(domain, config, judge)
    if not records:
        logger.info("score.skip domain=%s judge=%s (no records)", domain, judge)
        return
    domain_cfg = get_domain(domain)
    for pool in POOLS:
        lb = build_score_pool_leaderboard(
            config=config, domain=domain_cfg, judge_profile=judge,
            records=records, pool=pool,
        )
        write_json_atomic(score_pool_leaderboard_path(domain, config, judge, pool), lb)
        write_text_atomic(
            score_pool_leaderboard_path(domain, config, judge, pool, suffix="md"),
            render_leaderboard_md(lb),
        )
        logger.info(
            "score.ok domain=%s judge=%s pool=%s n_models=%d n_obs=%d",
            domain, judge, pool, lb.metadata.n_models, lb.metadata.n_observations,
        )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--domains", type=_csv, default=",".join(ALL_DOMAINS),
        help="Comma-list of domains; default all 6.",
    )
    parser.add_argument("--configs", type=_csv, default=None,
                        help="Comma-list of construction profiles to limit to.")
    parser.add_argument("--judges", type=_csv, default=None,
                        help="Comma-list of judge profiles to limit to.")
    parser.add_argument("--skip-arena", action="store_true")
    parser.add_argument("--skip-score", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    configure_runtime()
    domains = tuple(d for d in args.domains if d in ALL_DOMAINS)
    if not domains:
        logger.warning("no valid domains selected; exiting")
        return 1

    if not args.skip_arena:
        runs = _filter_runs(_discover_arena_runs(domains), configs=args.configs, judges=args.judges)
        logger.info("arena.runs total=%d", len(runs))
        for domain, config, judge in runs:
            rebuild_arena(domain=domain, config=config, judge=judge)

    if not args.skip_score:
        runs = _filter_runs(_discover_score_runs(domains), configs=args.configs, judges=args.judges)
        logger.info("score.runs total=%d", len(runs))
        for domain, config, judge in runs:
            rebuild_score(domain=domain, config=config, judge=judge)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
