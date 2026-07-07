"""Generic stage runner: scanner + promoter + worker pool with infinite retry-with-backoff."""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("hypo.orchestrator.stage")


@dataclass(frozen=True)
class StageItem:
    """Atomic work unit; key is the dedup/retry handle."""
    domain: str
    key: str
    short_label: str
    payload: Any


@dataclass
class StageConfig:
    """Per-stage tunables; orchestrator passes one per supervisor."""
    backoff_factor: float = 2.0
    backoff_initial: float = 30.0
    backoff_max: float = 1920.0
    concurrency: int = 4
    jitter: float = 0.3
    name: str = "stage"
    poll_interval: float = 10.0
    queue_maxsize: int = 0          # 0 = unbounded; supervisors usually pass concurrency × 4
    short_retry_after_crash: float = 30.0
    task_timeout: float = 300.0


def _jittered_backoff(attempt: int, cfg: StageConfig) -> float:
    """Exponential backoff with ±jitter; attempt is 1-based (first failure → 1)."""
    base = min(cfg.backoff_initial * (cfg.backoff_factor ** (attempt - 1)), cfg.backoff_max)
    factor = random.uniform(1.0 - cfg.jitter, 1.0 + cfg.jitter)
    return max(1.0, base * factor)


class StageObserver:
    """Stage notifies an observer (e.g., metrics sink) of lifecycle events. No-op default."""
    def task_started(self, stage: str, domain: str) -> None: ...
    def task_done(self, stage: str, domain: str, duration: float) -> None: ...
    def task_retry(self, stage: str, domain: str, label: str, attempt: int, delay: float, error: str) -> None: ...
    def stage_pending(self, stage: str, domain: str, todo: int) -> None: ...


@dataclass
class _StageState:
    """Bundles mutable per-stage state for clear ownership."""
    attempts: dict[str, int] = field(default_factory=dict)
    in_progress: int = 0
    ready_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    scheduled: dict[str, tuple[float, StageItem]] = field(default_factory=dict)
    seen: set[str] = field(default_factory=set)


class Stage:
    """Generic supervisor: scan → enqueue → workers process → infinite retry-with-backoff.

    Items that fail go back into the scheduled dict with jittered exponential backoff
    (capped at backoff_max), and the promoter releases them once retry_at expires. There
    is NO retry cap: every failed item keeps trying forever. To stop, the operator
    Ctrl+Cs the orchestrator. To prune a permanently-bad item, fix the underlying issue
    (e.g., remove a corrupt source file) and let the scanner stop yielding it.
    """

    def __init__(
        self,
        *,
        cfg: StageConfig,
        observer: StageObserver,
        scan_fn: Callable[[], Iterable[StageItem]],
        upstream_done: asyncio.Event,
        work_fn: Callable[[StageItem], Awaitable[None]],
    ):
        self.cfg = cfg
        self.observer = observer
        self.scan_fn = scan_fn
        self.upstream_done = upstream_done
        self.work_fn = work_fn

        self.done = asyncio.Event()
        self.state = _StageState(
            ready_queue=asyncio.Queue(maxsize=cfg.queue_maxsize),
        )
        self._scanner_task: asyncio.Task | None = None

    def _is_drained(self) -> bool:
        s = self.state
        return (
            s.ready_queue.empty()
            and not s.scheduled
            and s.in_progress == 0
        )

    async def _scanner(self) -> None:
        cfg = self.cfg
        s = self.state
        while True:
            try:
                pending = list(self.scan_fn())
            except Exception:
                logger.exception("stage=%s scan_fn raised", cfg.name)
                pending = []

            for item in pending:
                if item.key in s.seen:
                    continue
                s.seen.add(item.key)
                # Notify todo BEFORE put so the worker's task_started can decrement it
                # consistently; with a bounded queue, deferring this until end-of-pass
                # would leave the dashboard at todo=0 for hours.
                self.observer.stage_pending(cfg.name, item.domain, 1)
                await s.ready_queue.put(item)            # blocks if queue is full → backpressure

            if self.upstream_done.is_set() and self._is_drained():
                # Inject sentinels (one per worker) and exit.
                for _ in range(cfg.concurrency):
                    await s.ready_queue.put(None)
                return

            await asyncio.sleep(cfg.poll_interval)

    async def _promoter(self) -> None:
        s = self.state
        while True:
            now = time.time()
            ready_keys = [k for k, (retry_at, _) in s.scheduled.items() if retry_at <= now]
            for key in ready_keys:
                _, item = s.scheduled.pop(key)
                await s.ready_queue.put(item)
            if (
                not s.scheduled
                and self._scanner_task is not None
                and self._scanner_task.done()
            ):
                return
            await asyncio.sleep(1.0)

    async def _worker(self) -> None:
        cfg = self.cfg
        s = self.state
        while True:
            item = await s.ready_queue.get()
            if item is None:
                s.ready_queue.task_done()
                return
            s.in_progress += 1
            self.observer.task_started(cfg.name, item.domain)
            t0 = time.time()
            try:
                try:
                    await asyncio.wait_for(self.work_fn(item), timeout=cfg.task_timeout)
                    duration = time.time() - t0
                    s.attempts.pop(item.key, None)
                    self.observer.task_done(cfg.name, item.domain, duration)
                except (asyncio.TimeoutError, Exception) as exc:
                    n = s.attempts.get(item.key, 0) + 1
                    s.attempts[item.key] = n
                    error_str = f"{type(exc).__name__}: {str(exc)[:200]}"
                    delay = _jittered_backoff(n, cfg)
                    s.scheduled[item.key] = (time.time() + delay, item)
                    self.observer.task_retry(cfg.name, item.domain, item.short_label, n, delay, error_str)
                    logger.info(
                        "retry stage=%s key=%s attempt=%d in=%.1fs err=%s",
                        cfg.name, item.key, n, delay, error_str,
                    )
            except BaseException:
                # Worker-level safety net: never lose an item to a bug.
                logger.exception("stage=%s worker fatal key=%s", cfg.name, item.key)
                s.scheduled[item.key] = (time.time() + cfg.short_retry_after_crash, item)
                raise
            finally:
                s.in_progress -= 1
                s.ready_queue.task_done()

    async def run(self) -> None:
        """Run scanner + promoter + N workers; resolve when fully drained.

        Note: with infinite retry, _is_drained() only becomes true once every item has
        succeeded. A permanently-failing item keeps the stage running forever; operator
        must Ctrl+C to stop.
        """
        try:
            self._scanner_task = asyncio.create_task(self._scanner(), name=f"{self.cfg.name}.scanner")
            promoter_task = asyncio.create_task(self._promoter(), name=f"{self.cfg.name}.promoter")
            worker_tasks = [
                asyncio.create_task(self._worker(), name=f"{self.cfg.name}.worker[{i}]")
                for i in range(self.cfg.concurrency)
            ]
            await self._scanner_task
            await asyncio.gather(*worker_tasks)
            await promoter_task
        finally:
            self.done.set()
