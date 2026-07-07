"""Streaming JSONL I/O helpers for every pipeline-stage artifact."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, TypeVar

from pydantic import BaseModel

from .paths import (
    arena_leaderboard_path,
    arena_matches_path,
    cases_path,
    score_leaderboard_path,
    score_records_path,
    source_dir,
    submission_path,
    submissions_glob,
)
from .schema import (
    ArenaMatch,
    BenchmarkCase,
    Leaderboard,
    Method,
    Mode,
    ScoreRecord,
    SourceRecord,
    Submission,
)

T = TypeVar("T", bound=BaseModel)


# ---- generic JSONL ----

def iter_jsonl(path: Path, model: type[T]) -> Iterator[T]:
    """Yield model instances parsed from each non-empty line of a JSONL file."""
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield model.model_validate_json(line)


def append_jsonl(path: Path, item: BaseModel) -> None:
    """Append one model as a JSON line; create the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(item.model_dump_json() + "\n")


def write_json(path: Path, item: BaseModel, *, indent: int = 2) -> None:
    """Write a single model as a pretty-printed JSON file (overwrite)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(item.model_dump_json(indent=indent), encoding="utf-8")


def existing_ids(path: Path) -> set[str]:
    """Set of `id`/`case_id` already present in a JSONL file (for resume support)."""
    ids: set[str] = set()
    if not path.exists():
        return ids
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = row.get("id") or row.get("case_id")
            if key:
                ids.add(key)
    return ids


# ---- source records ----

def source_metadata_path(domain: str) -> Path:
    """Convention: source records live in source/metadata.jsonl under each domain."""
    return source_dir(domain) / "metadata.jsonl"


def load_sources(domain: str) -> list[SourceRecord]:
    """Read all SourceRecord rows for a domain."""
    return list(iter_jsonl(source_metadata_path(domain), SourceRecord))


# ---- benchmark cases (stage 1) ----

def append_case(domain: str, config: str, case: BenchmarkCase) -> None:
    append_jsonl(cases_path(domain, config), case)


def load_cases(domain: str, config: str, *, only_passed: bool = False) -> list[BenchmarkCase]:
    """All BenchmarkCase rows for a (domain, construction profile); optionally filter to passed."""
    cases = list(iter_jsonl(cases_path(domain, config), BenchmarkCase))
    if only_passed:
        cases = [c for c in cases if c.quality.passed]
    return cases


# ---- submissions (stage 2) ----

def append_submission(domain: str, mode: Mode, config: str, profile: str, sub: Submission) -> None:
    append_jsonl(submission_path(domain, mode, config, profile), sub)


def load_submissions(domain: str, mode: Mode, config: str, profile: str) -> list[Submission]:
    return list(iter_jsonl(submission_path(domain, mode, config, profile), Submission))


def load_all_submissions(
    domain: str,
    mode: Mode,
    config: str,
) -> dict[str, list[Submission]]:
    """{generation_profile: [Submission]} for every generation profile under (domain, mode, config)."""
    out: dict[str, list[Submission]] = {}
    for path in submissions_glob(domain, mode, config):
        stem = path.stem
        if "+" not in stem:
            continue
        profile = stem.split("+", 1)[1]
        out[profile] = list(iter_jsonl(path, Submission))
    return out


# ---- arena (stage 3) ----

def append_arena_match(domain: str, config: str, judge: str, match: ArenaMatch) -> None:
    append_jsonl(arena_matches_path(domain, config, judge), match)


def load_arena_matches(domain: str, config: str, judge: str) -> list[ArenaMatch]:
    return list(iter_jsonl(arena_matches_path(domain, config, judge), ArenaMatch))


def write_arena_leaderboard(domain: str, config: str, judge: str, leaderboard: Leaderboard) -> None:
    write_json(arena_leaderboard_path(domain, config, judge), leaderboard)


# ---- score (stage 3) ----

def append_score_record(domain: str, config: str, judge: str, record: ScoreRecord) -> None:
    append_jsonl(score_records_path(domain, config, judge), record)


def load_score_records(domain: str, config: str, judge: str) -> list[ScoreRecord]:
    return list(iter_jsonl(score_records_path(domain, config, judge), ScoreRecord))


def write_score_leaderboard(domain: str, config: str, judge: str, leaderboard: Leaderboard) -> None:
    write_json(score_leaderboard_path(domain, config, judge), leaderboard)


# ---- arena/score match-key set (resume) ----

def existing_arena_pair_keys(domain: str, config: str, judge: str) -> set[tuple[str, str, str]]:
    """Set of (case_id, model_a, model_b) already judged, for resume."""
    keys: set[tuple[str, str, str]] = set()
    for match in iter_jsonl(arena_matches_path(domain, config, judge), ArenaMatch):
        keys.add((match.case_id, match.model_a, match.model_b))
    return keys


def existing_score_keys(domain: str, config: str, judge: str) -> set[tuple[str, str]]:
    """Set of (case_id, model) already scored, for resume."""
    keys: set[tuple[str, str]] = set()
    for r in iter_jsonl(score_records_path(domain, config, judge), ScoreRecord):
        keys.add((r.case_id, r.model))
    return keys


# ---- re-export (so callers only import basics.io) ----

__all__ = [
    "Method",
    "Mode",
    "append_arena_match",
    "append_case",
    "append_jsonl",
    "append_score_record",
    "append_submission",
    "existing_arena_pair_keys",
    "existing_ids",
    "existing_score_keys",
    "iter_jsonl",
    "load_all_submissions",
    "load_arena_matches",
    "load_cases",
    "load_score_records",
    "load_sources",
    "load_submissions",
    "source_metadata_path",
    "write_arena_leaderboard",
    "write_json",
    "write_score_leaderboard",
]
