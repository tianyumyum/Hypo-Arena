"""Path generators for every artifact under artifacts/."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from .schema import Method, Mode, Pool

ARTIFACTS_ROOT = Path(__file__).parents[1] / "artifacts"

Suffix = Literal["json", "md"]


# ---- stage 1: source + cases ----

def source_dir(domain: str) -> Path:
    """Where raw input documents for this domain live."""
    return ARTIFACTS_ROOT / domain / "source"


def cases_path(domain: str, config: str) -> Path:
    """Construction output: one JSONL per (domain, construction profile)."""
    return ARTIFACTS_ROOT / domain / "cases" / f"{config}.jsonl"


# ---- stage 2: submissions ----

def submission_path(domain: str, mode: Mode, config: str, profile: str) -> Path:
    """One JSONL per (domain, mode, construction config, generation profile)."""
    return ARTIFACTS_ROOT / domain / "submissions" / mode / f"{config}+{profile}.jsonl"


def submissions_glob(domain: str, mode: Mode, config: str) -> list[Path]:
    """All generation profiles' submission files for one (domain, mode, config)."""
    base = ARTIFACTS_ROOT / domain / "submissions" / mode
    return sorted(base.glob(f"{config}+*.jsonl"))


# ---- stage 3: arena ----

def arena_matches_path(domain: str, config: str, judge: str) -> Path:
    """Per-match record stream from arena judging."""
    return ARTIFACTS_ROOT / domain / "results" / f"{config}.{judge}.arena.matches.jsonl"


def arena_leaderboard_path(domain: str, config: str, judge: str, suffix: Suffix = "json") -> Path:
    """Aggregated arena leaderboard (BTD ratings) — pool-agnostic legacy path."""
    return ARTIFACTS_ROOT / domain / "results" / f"{config}.{judge}.arena.leaderboard.{suffix}"


def arena_pool_leaderboard_path(
    domain: str, config: str, judge: str, pool: Pool, suffix: Suffix = "json",
) -> Path:
    """Pool-filtered arena leaderboard: one of baseline / agent / full."""
    return (
        ARTIFACTS_ROOT / domain / "results"
        / f"{config}.{judge}.arena.{pool}.leaderboard.{suffix}"
    )


# ---- stage 3: score ----

def score_records_path(domain: str, config: str, judge: str) -> Path:
    """Per-(case, model) absolute scoring records."""
    return ARTIFACTS_ROOT / domain / "results" / f"{config}.{judge}.score.jsonl"


def score_leaderboard_path(domain: str, config: str, judge: str, suffix: Suffix = "json") -> Path:
    """Aggregated score leaderboard (dimension-averaged) — pool-agnostic legacy path."""
    return ARTIFACTS_ROOT / domain / "results" / f"{config}.{judge}.score.leaderboard.{suffix}"


def score_pool_leaderboard_path(
    domain: str, config: str, judge: str, pool: Pool, suffix: Suffix = "json",
) -> Path:
    """Pool-filtered score leaderboard: one of baseline / agent / full."""
    return (
        ARTIFACTS_ROOT / domain / "results"
        / f"{config}.{judge}.score.{pool}.leaderboard.{suffix}"
    )


# ---- cross-domain summary ----

def summary_path(method: Method, config: str, judge: str, suffix: Suffix = "json") -> Path:
    """Cross-domain leaderboard summary."""
    return ARTIFACTS_ROOT / "_summary" / f"{config}.{judge}.{method}.summary.{suffix}"
