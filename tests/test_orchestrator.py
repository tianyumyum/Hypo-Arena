"""Unit tests for the orchestrator: stage runner semantics, raw keys, atomic write."""

from __future__ import annotations

import asyncio
import json

import pytest
from pydantic import BaseModel

from orchestrator.atomic import write_json_atomic
from orchestrator.keys import (
    invalidate_cache,
    raw_arena_pair_keys,
    raw_existing_ids,
    raw_score_keys,
)
from orchestrator.stage import (
    Stage,
    StageConfig,
    StageItem,
    StageObserver,
    _jittered_backoff,
)


class _RecordingObserver(StageObserver):
    """Test double capturing all lifecycle calls for assertion."""

    def __init__(self) -> None:
        self.started: list[tuple[str, str]] = []
        self.done: list[tuple[str, str, float]] = []
        self.retries: list[tuple[str, str, int, float, str]] = []
        self.pending: list[tuple[str, str, int]] = []

    def task_started(self, stage: str, domain: str) -> None:
        self.started.append((stage, domain))

    def task_done(self, stage: str, domain: str, duration: float) -> None:
        self.done.append((stage, domain, duration))

    def task_retry(self, stage, domain, label, attempt, delay, error):
        self.retries.append((stage, domain, attempt, delay, error))

    def stage_pending(self, stage, domain, todo):
        self.pending.append((stage, domain, todo))


# ---- Raw keys ----

def test_raw_existing_ids(tmp_path):
    path = tmp_path / "x.jsonl"
    path.write_text('{"id": "a"}\n{"case_id": "b"}\n{}\n', encoding="utf-8")
    invalidate_cache()
    keys = raw_existing_ids(path)
    assert keys == {"a", "b"}


def test_raw_arena_pair_keys(tmp_path):
    path = tmp_path / "arena.jsonl"
    path.write_text(
        '{"case_id": "c1", "model_a": "X", "model_b": "Y"}\n'
        '{"case_id": "c1", "model_a": "Y", "model_b": "X"}\n'
        '{"case_id": "c2", "model_a": "X", "model_b": "Z"}\n',
        encoding="utf-8",
    )
    invalidate_cache()
    keys = raw_arena_pair_keys(path)
    assert keys == {
        ("c1", frozenset(["X", "Y"])),
        ("c2", frozenset(["X", "Z"])),
    }


def test_raw_score_keys(tmp_path):
    path = tmp_path / "score.jsonl"
    path.write_text('{"case_id": "c1", "model": "m1"}\n{"case_id": "c2", "model": "m2"}\n', encoding="utf-8")
    invalidate_cache()
    keys = raw_score_keys(path)
    assert keys == {("c1", "m1"), ("c2", "m2")}


# ---- Atomic write ----

def test_write_json_atomic(tmp_path):
    class M(BaseModel):
        k: int
        v: str

    target = tmp_path / "out.json"
    write_json_atomic(target, M(k=1, v="a"))
    assert target.exists()
    data = json.loads(target.read_text(encoding="utf-8"))
    assert data == {"k": 1, "v": "a"}
    assert not (target.with_suffix(".json.tmp").exists())


# ---- Jittered backoff ----

def test_jittered_backoff_within_bounds():
    cfg = StageConfig(
        backoff_initial=30.0, backoff_factor=2.0, backoff_max=1920.0, jitter=0.3,
    )
    # attempt 1 base = 30s; with 30% jitter, 21s ≤ delay ≤ 39s
    for _ in range(100):
        d = _jittered_backoff(1, cfg)
        assert 21.0 <= d <= 39.0


def test_backoff_caps_at_max():
    cfg = StageConfig(
        backoff_initial=30.0, backoff_factor=2.0, backoff_max=60.0, jitter=0.0,
    )
    # attempt 10 base = 30 * 2^9 = 15360, but capped at 60
    assert _jittered_backoff(10, cfg) == 60.0


# ---- Stage end-to-end ----

async def _run_stage_to_completion(stage: Stage, timeout: float = 5.0) -> None:
    await asyncio.wait_for(stage.run(), timeout=timeout)


@pytest.mark.asyncio
async def test_stage_completes_all_items():
    processed: list[str] = []

    def scan():
        for i in range(3):
            yield StageItem(domain="d1", key=f"k{i}", short_label=f"item{i}", payload=i)

    seen_ever: set[str] = set()

    async def work(item: StageItem):
        processed.append(item.key)
        seen_ever.add(item.key)

    def scan_wrapper():
        # Only yield items not already processed (so once done, scan emits nothing new)
        for item in scan():
            if item.key not in seen_ever:
                yield item

    observer = _RecordingObserver()
    upstream = asyncio.Event()
    upstream.set()

    stage = Stage(
        cfg=StageConfig(
            name="test", concurrency=2, task_timeout=2.0,
            poll_interval=0.2, queue_maxsize=10,
        ),
        observer=observer,
        scan_fn=scan_wrapper,
        upstream_done=upstream,
        work_fn=work,
    )
    await _run_stage_to_completion(stage)

    assert sorted(processed) == ["k0", "k1", "k2"]
    assert len(observer.done) == 3
    assert len(observer.retries) == 0


@pytest.mark.asyncio
async def test_stage_retries_then_succeeds():
    attempts_seen: dict[str, int] = {"k0": 0}

    async def work(item: StageItem):
        attempts_seen[item.key] += 1
        if attempts_seen[item.key] < 3:
            raise RuntimeError("transient")

    def scan():
        yield StageItem(domain="d1", key="k0", short_label="item", payload=None)

    observer = _RecordingObserver()
    upstream = asyncio.Event()
    upstream.set()

    stage = Stage(
        cfg=StageConfig(
            name="test", concurrency=1, task_timeout=2.0,
            poll_interval=0.05, queue_maxsize=5,
            backoff_initial=0.05, backoff_factor=1.5, backoff_max=0.5, jitter=0.0,
        ),
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream,
        work_fn=work,
    )
    await _run_stage_to_completion(stage, timeout=10.0)

    assert attempts_seen["k0"] == 3
    assert len(observer.retries) == 2
    assert len(observer.done) == 1


@pytest.mark.asyncio
async def test_stage_retries_indefinitely_until_success():
    """Without a max-attempts cap, an item that finally succeeds at attempt N is recorded."""
    attempts_seen: dict[str, int] = {"k0": 0}

    async def work(item: StageItem):
        attempts_seen[item.key] += 1
        if attempts_seen[item.key] < 7:                        # > old default of 8 to prove no cap
            raise RuntimeError("flaky")

    def scan():
        yield StageItem(domain="d1", key="k0", short_label="item", payload=None)

    observer = _RecordingObserver()
    upstream = asyncio.Event()
    upstream.set()

    stage = Stage(
        cfg=StageConfig(
            name="test", concurrency=1, task_timeout=1.0,
            poll_interval=0.02, queue_maxsize=5,
            backoff_initial=0.02, backoff_factor=1.0, backoff_max=0.05, jitter=0.0,
        ),
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream,
        work_fn=work,
    )
    await _run_stage_to_completion(stage, timeout=10.0)

    assert attempts_seen["k0"] == 7
    assert len(observer.done) == 1
    assert len(observer.retries) == 6


@pytest.mark.asyncio
async def test_stage_timeout_treated_as_retry():
    """asyncio.TimeoutError from wait_for must trigger a retry (not crash the worker)."""
    attempts: dict[str, int] = {"k0": 0}

    async def work(item: StageItem):
        attempts[item.key] += 1
        if attempts[item.key] == 1:
            await asyncio.sleep(10)                           # exceeds task_timeout below

    def scan():
        yield StageItem(domain="d1", key="k0", short_label="item", payload=None)

    observer = _RecordingObserver()
    upstream = asyncio.Event()
    upstream.set()

    stage = Stage(
        cfg=StageConfig(
            name="test", concurrency=1, task_timeout=0.1,
            poll_interval=0.05, queue_maxsize=5,
            backoff_initial=0.02, backoff_factor=1.2, backoff_max=0.1, jitter=0.0,
        ),
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream,
        work_fn=work,
    )
    await _run_stage_to_completion(stage, timeout=5.0)

    assert len(observer.retries) == 1
    assert "TimeoutError" in observer.retries[0][4]            # error_str field
    assert len(observer.done) == 1


@pytest.mark.asyncio
async def test_stage_exits_only_when_scheduled_drained():
    """Scanner must not inject sentinels while retry-pending items remain."""
    counter = {"n": 0}

    async def work(item: StageItem):
        counter["n"] += 1
        if counter["n"] < 3:
            raise RuntimeError("flaky")

    def scan():
        yield StageItem(domain="d1", key="k0", short_label="only", payload=None)

    observer = _RecordingObserver()
    upstream = asyncio.Event()
    upstream.set()

    stage = Stage(
        cfg=StageConfig(
            name="test", concurrency=1, task_timeout=1.0,
            poll_interval=0.05, queue_maxsize=5,
            backoff_initial=0.1, backoff_factor=1.0, backoff_max=0.1, jitter=0.0,
        ),
        observer=observer,
        scan_fn=scan,
        upstream_done=upstream,
        work_fn=work,
    )
    await _run_stage_to_completion(stage, timeout=5.0)

    # Must have completed eventually, so scanner waited for retries to drain
    assert len(observer.done) == 1


# ---- Pool-aware leaderboard rebuild (Stage E) ----

def test_rebuild_arena_leaderboard_writes_three_pool_files(tmp_path, monkeypatch):
    """_rebuild_arena_leaderboard must emit baseline / agent / full leaderboards."""
    import basics.paths as paths_module
    monkeypatch.setattr(paths_module, "ARTIFACTS_ROOT", tmp_path)

    from basics.io import append_arena_match
    from basics.schema import ArenaMatch, JudgeVerdict
    from orchestrator.supervisors import (
        OrchestratorConfig,
        _rebuild_arena_leaderboard,
    )

    domain, config, judge = "biomedical_science", "gpt-5.4-test", "mimo-v2-pro"

    labels = ["reference", "baseline:m1", "baseline:m2", "agent:m1", "agent:m2"]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            for case_id in ("c1", "c2"):
                match = ArenaMatch(
                    case_id=case_id,
                    model_a=labels[i],
                    model_b=labels[j],
                    judge=judge,
                    forward=JudgeVerdict(winner="a", score=0.75),
                    reverse=JudgeVerdict(winner="b", score=0.25),
                )
                append_arena_match(domain, config, judge, match)

    cfg = OrchestratorConfig(
        construction_profile=config,
        domains=(domain,),
        judge_profiles=(judge,),
    )
    _rebuild_arena_leaderboard(domain, judge, cfg)

    results_dir = tmp_path / domain / "results"
    for pool in ("baseline", "agent", "full"):
        json_path = results_dir / f"{config}.{judge}.arena.{pool}.leaderboard.json"
        md_path = results_dir / f"{config}.{judge}.arena.{pool}.leaderboard.md"
        assert json_path.exists(), f"missing {json_path}"
        assert md_path.exists(), f"missing {md_path}"

        lb = json.loads(json_path.read_text(encoding="utf-8"))
        assert lb["metadata"]["pool"] == pool
        models = {e["model"] for e in lb["rankings"]}
        assert "reference" in models                            # always retained
        if pool == "baseline":
            assert "baseline:m1" in models and "baseline:m2" in models
            assert "agent:m1" not in models and "agent:m2" not in models
        elif pool == "agent":
            assert "agent:m1" in models and "agent:m2" in models
            assert "baseline:m1" not in models and "baseline:m2" not in models
        else:
            assert models == set(labels)


def test_backfill_leaderboard_markdown_skips_legacy_files(tmp_path, monkeypatch):
    """Pre-Stage-E `*.arena.leaderboard.json` files must NOT regenerate stale .md companions."""
    import basics.paths as paths_module
    monkeypatch.setattr(paths_module, "ARTIFACTS_ROOT", tmp_path)

    import orchestrator
    monkeypatch.setattr(orchestrator, "ARTIFACTS_ROOT", tmp_path)

    results = tmp_path / "biomedical_science" / "results"
    results.mkdir(parents=True)

    # Legacy file (pool-agnostic) — should be skipped.
    legacy_json = results / "gpt-5.4.mimo-v2-pro.arena.leaderboard.json"
    legacy_json.write_text(
        json.dumps({
            "metadata": {
                "config": "gpt-5.4", "domain": "biomedical_science",
                "judge": "mimo-v2-pro", "method": "arena",
                "n_models": 1, "n_observations": 1,
            },
            "rankings": [{"rank": 1, "model": "x", "rating": 1500.0,
                          "n_observations": 1, "breakdown": {}}],
        }),
        encoding="utf-8",
    )

    # Pool-aware file — should produce a fresh .md.
    pool_json = results / "gpt-5.4.mimo-v2-pro.arena.full.leaderboard.json"
    pool_json.write_text(legacy_json.read_text(encoding="utf-8"), encoding="utf-8")

    rebuilt, skipped = orchestrator._backfill_leaderboard_markdown()
    assert rebuilt == 1                                    # only the pool file
    assert skipped == 1                                    # the legacy file
    assert (results / "gpt-5.4.mimo-v2-pro.arena.full.leaderboard.md").exists()
    assert not (results / "gpt-5.4.mimo-v2-pro.arena.leaderboard.md").exists()


def test_rebuild_score_leaderboard_writes_three_pool_files(tmp_path, monkeypatch):
    """_rebuild_score_leaderboard must emit baseline / agent / full leaderboards."""
    import basics.paths as paths_module
    monkeypatch.setattr(paths_module, "ARTIFACTS_ROOT", tmp_path)

    from basics.io import append_score_record
    from basics.schema import ScoreRecord
    from orchestrator.supervisors import (
        OrchestratorConfig,
        _rebuild_score_leaderboard,
    )

    domain, config, judge = "biomedical_science", "gpt-5.4-test", "mimo-v2-pro"
    for label in ("reference", "baseline:m1", "agent:m1"):
        record = ScoreRecord(
            case_id="c1",
            model=label,
            judge=judge,
            pair_scores=[{"grounding": 4.0, "insight": 4.0, "justification": 4.0}],
        )
        append_score_record(domain, config, judge, record)

    cfg = OrchestratorConfig(
        construction_profile=config,
        domains=(domain,),
        judge_profiles=(judge,),
    )
    _rebuild_score_leaderboard(domain, judge, cfg)

    results_dir = tmp_path / domain / "results"
    for pool in ("baseline", "agent", "full"):
        json_path = results_dir / f"{config}.{judge}.score.{pool}.leaderboard.json"
        assert json_path.exists(), f"missing {json_path}"
        lb = json.loads(json_path.read_text(encoding="utf-8"))
        assert lb["metadata"]["pool"] == pool
        models = {e["model"] for e in lb["rankings"]}
        assert "reference" in models
