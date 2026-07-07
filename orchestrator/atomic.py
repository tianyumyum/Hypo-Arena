"""Atomic JSON / text writes via tmp file + os.replace."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pydantic import BaseModel


def write_json_atomic(path: Path, item: BaseModel, *, indent: int = 2) -> None:
    """Write a Pydantic model to JSON atomically: write to .tmp then os.replace into place."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(item.model_dump(mode="json"), f, ensure_ascii=False, indent=indent)
    os.replace(tmp, path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write plain text atomically; used for human-readable companions (.md)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
