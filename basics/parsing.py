"""Defensive JSON-from-text extraction for models that can't / don't honor strict JSON schema.

Used by both evaluation (judges) and generation (writers / skill_selector) when the
target profile is marked `supports_response_format=False`. Helpers mirror
hypoarena/utils/utils.py.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


def _strip_code_fences(text: str) -> str:
    """Remove markdown code fences (```json ... ```) wrapping the JSON body."""
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 2:
            return "\n".join(lines[1:-1]).strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped)
    if match:
        return match.group(1).strip()
    return stripped


def _escape_newlines_in_strings(value: str) -> str:
    """Escape literal newlines that landed inside JSON string values."""
    out: list[str] = []
    in_string = False
    backslashes = 0
    for char in value:
        if char == "\\":
            out.append(char)
            backslashes += 1
            continue
        if char == '"' and backslashes % 2 == 0:
            in_string = not in_string
        if char == "\n" and in_string:
            out.append("\\n")
        elif char == "\r" and in_string:
            out.append("\\r")
        else:
            out.append(char)
        backslashes = 0
    return "".join(out)


def _fix_invalid_escapes(text: str) -> str:
    """Replace illegal JSON escape sequences (\\p, \\d, \\s, …) by escaping the backslash."""
    valid = set(r'"\\/bfnrtu')
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in valid:
                out.append(ch)
                out.append(nxt)
                i += 2
                continue
            out.append("\\\\")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _fix_trailing_comma(text: str) -> str:
    """Remove trailing commas before closing } or ]."""
    return re.sub(r",\s*([}\]])", r"\1", text)


def extract_json(text: str) -> Any:
    """Pull a JSON value out of LLM text with multi-stage tolerant fallback.

    Handles markdown fences, trailing commas, invalid escapes, extra trailing content.
    Raises json.JSONDecodeError if nothing parses.
    """
    candidate = _strip_code_fences(text)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end >= start:
        candidate = candidate[start:end + 1]

    decoder = json.JSONDecoder()

    def _try(t: str) -> Any:
        try:
            return json.loads(t)
        except json.JSONDecodeError:
            parsed, _ = decoder.raw_decode(t)
            return parsed

    transforms = [
        lambda t: t,
        _fix_invalid_escapes,
        _escape_newlines_in_strings,
        _fix_trailing_comma,
        lambda t: _fix_trailing_comma(_fix_invalid_escapes(t)),
        lambda t: _escape_newlines_in_strings(_fix_invalid_escapes(t)),
    ]
    for transform in transforms:
        try:
            return _try(transform(candidate))
        except (json.JSONDecodeError, ValueError):
            continue
    parsed, _ = decoder.raw_decode(candidate)                  # final, will raise if hopeless
    return parsed


def coerce_to_model(value: Any, model_class: type[T]) -> T:
    """Accept either an already-validated model instance or raw text; return model_class.

    - Structured-output mode: SDK already returns a `model_class` instance — returned as is.
    - Text mode: parse string → JSON → Pydantic-validate into `model_class`.
    """
    if isinstance(value, model_class):
        return value
    if isinstance(value, str):
        payload = extract_json(value)
        return model_class.model_validate(payload)
    raise TypeError(
        f"Cannot coerce {type(value).__name__} to {model_class.__name__}; "
        f"expected the model itself or a string."
    )
