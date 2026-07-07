"""Local-only tracing: structured JSONL export + compact stderr log; no remote upload."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from agents import set_trace_processors
from agents.tracing import TracingProcessor
from agents.tracing.processor_interface import TracingExporter
from agents.tracing.processors import BatchTraceProcessor
from agents.tracing.spans import Span
from agents.tracing.traces import Trace

logger = logging.getLogger("hypo.agent")


class JsonlExporter(TracingExporter):
    """Append every Trace and Span as a JSON line to a local file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, items: list[Trace | Span[Any]]) -> None:
        lines: list[str] = []
        for item in items:
            payload = item.export()
            if payload is None:
                continue
            kind = "trace" if isinstance(item, Trace) else "span"
            lines.append(json.dumps({"object": kind, **payload}, default=str, ensure_ascii=False))
        if not lines:
            return
        with self._lock, self._path.open("a") as fh:
            for line in lines:
                fh.write(line + "\n")


class StderrLogProcessor(TracingProcessor):
    """Emit one compact log line per finished span (real-time, no batching)."""

    def on_trace_start(self, trace: Trace) -> None:
        logger.info("trace.start name=%s id=%s", trace.name, trace.trace_id)

    def on_trace_end(self, trace: Trace) -> None:
        logger.info("trace.end name=%s", trace.name)

    def on_span_start(self, span: Span[Any]) -> None:
        pass

    def on_span_end(self, span: Span[Any]) -> None:
        sd = span.span_data
        kind = type(sd).__name__.removesuffix("SpanData").lower()
        fields = _format_span_fields(kind, sd)
        suffix = " ".join(fields)
        logger.info("span.end type=%s%s", kind, f" {suffix}" if suffix else "")

    def shutdown(self) -> None:
        pass

    def force_flush(self) -> None:
        pass


def _format_span_fields(kind: str, sd: Any) -> list[str]:
    """Pick the salient fields per span kind for a compact one-line summary."""
    if kind == "agent":
        return [f"name={getattr(sd, 'name', '?')!r}"]
    if kind == "generation":
        usage = getattr(sd, "usage", None) or {}
        return [
            f"model={getattr(sd, 'model', '?')!r}",
            f"in={usage.get('input_tokens', 0)}",
            f"out={usage.get('output_tokens', 0)}",
        ]
    if kind == "function":
        return [f"name={getattr(sd, 'name', '?')!r}"]
    if kind == "handoff":
        return [
            f"from={getattr(sd, 'from_agent', '?')!r}",
            f"to={getattr(sd, 'to_agent', '?')!r}",
        ]
    if kind == "guardrail":
        return [
            f"name={getattr(sd, 'name', '?')!r}",
            f"triggered={getattr(sd, 'triggered', '?')}",
        ]
    return []


def configure_tracing(jsonl_path: Path | str | None = None) -> None:
    """Replace SDK default OpenAI export with local processors.

    By default only stderr compact log is active (cheap, bounded). Pass `jsonl_path`
    to enable per-span JSONL dump — but be aware the file grows ~3 MB/s under heavy
    pipeline load, so prefer timestamped paths and rotate manually.
    """
    processors: list = [StderrLogProcessor()]
    if jsonl_path:
        processors.insert(0, BatchTraceProcessor(exporter=JsonlExporter(jsonl_path)))
    set_trace_processors(processors)
