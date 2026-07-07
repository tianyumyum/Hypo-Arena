"""Raw-JSON dedup-key extraction from artifact JSONLs (no Pydantic, mtime-cached)."""

from __future__ import annotations

import json
from pathlib import Path

# (path, mtime) → cached key set; rescan only on mtime change.
_id_cache: dict[Path, tuple[float, set[str]]] = {}
_arena_cache: dict[Path, tuple[float, set[tuple[str, frozenset[str]]]]] = {}
_score_cache: dict[Path, tuple[float, set[tuple[str, str]]]] = {}


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


def _iter_records(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def raw_existing_ids(path: Path) -> set[str]:
    """IDs already present in a JSONL (case_id or id field). mtime-cached."""
    mtime = _mtime(path)
    cached = _id_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    keys: set[str] = set()
    for rec in _iter_records(path):
        key = rec.get("id") or rec.get("case_id")
        if key:
            keys.add(key)
    _id_cache[path] = (mtime, keys)
    return keys


def raw_arena_pair_keys(path: Path) -> set[tuple[str, frozenset[str]]]:
    """Arena pairs already judged: set of (case_id, frozenset({model_a, model_b})). mtime-cached."""
    mtime = _mtime(path)
    cached = _arena_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    keys: set[tuple[str, frozenset[str]]] = set()
    for rec in _iter_records(path):
        case_id = rec.get("case_id")
        a = rec.get("model_a")
        b = rec.get("model_b")
        if case_id and a and b:
            keys.add((case_id, frozenset([a, b])))
    _arena_cache[path] = (mtime, keys)
    return keys


def raw_score_keys(path: Path) -> set[tuple[str, str]]:
    """Score records already produced: set of (case_id, model). mtime-cached."""
    mtime = _mtime(path)
    cached = _score_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    keys: set[tuple[str, str]] = set()
    for rec in _iter_records(path):
        case_id = rec.get("case_id")
        model = rec.get("model")
        if case_id and model:
            keys.add((case_id, model))
    _score_cache[path] = (mtime, keys)
    return keys


def invalidate_cache(path: Path | None = None) -> None:
    """Force re-read on next call. Pass None to clear all caches."""
    if path is None:
        _id_cache.clear()
        _arena_cache.clear()
        _score_cache.clear()
    else:
        _id_cache.pop(path, None)
        _arena_cache.pop(path, None)
        _score_cache.pop(path, None)
