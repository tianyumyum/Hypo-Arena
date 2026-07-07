"""Process-wide runtime setup: stdlib logging + local-only tracing (JSONL + stderr)."""

from __future__ import annotations

import logging
import sys

from .tracing import configure_tracing

_CONFIGURED = False


def configure_runtime() -> None:
    """Idempotent: call once at process start."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("hypo")
    root.setLevel(logging.INFO)
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
    root.propagate = False

    configure_tracing()

    _CONFIGURED = True
