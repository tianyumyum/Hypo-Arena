"""Transport layer: AsyncOpenAI client factories per platform + cross-platform fallback Model."""

from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator
from typing import Literal

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
)

from agents.items import ModelResponse, TResponseStreamEvent
from agents.models.interface import Model

logger = logging.getLogger("hypo.agent")


# ---- platform clients ----

Platform = Literal["platform_a", "platform_b", "platform_c", "platform_d"]

_ENV_KEYS: dict[Platform, tuple[str, str]] = {
    "platform_a": ("PLATFORM_A_API_KEY", "PLATFORM_A_BASE_URL"),
    "platform_b": ("PLATFORM_B_API_KEY", "PLATFORM_B_BASE_URL"),
    "platform_c": ("PLATFORM_C_API_KEY", "PLATFORM_C_BASE_URL"),
    "platform_d": ("PLATFORM_D_API_KEY", "PLATFORM_D_BASE_URL"),
}

_clients: dict[Platform, AsyncOpenAI] = {}


def is_available(platform: Platform) -> bool:
    """True iff both env vars for this platform are set."""
    api_key_var, base_url_var = _ENV_KEYS[platform]
    return bool(os.environ.get(api_key_var)) and bool(os.environ.get(base_url_var))


def get_client(platform: Platform) -> AsyncOpenAI:
    """Return a cached AsyncOpenAI for the named platform."""
    cached = _clients.get(platform)
    if cached is not None:
        return cached

    api_key_var, base_url_var = _ENV_KEYS[platform]
    api_key = os.environ.get(api_key_var)
    base_url = os.environ.get(base_url_var)
    if not api_key or not base_url:
        missing = [v for v, val in [(api_key_var, api_key), (base_url_var, base_url)] if not val]
        raise RuntimeError(
            f"Platform {platform!r} not configured. Set: {', '.join(missing)}"
        )

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    _clients[platform] = client
    return client


# ---- cross-platform fallback Model ----

# Network + any HTTP status error: catch APIStatusError broadly so the FallbackModel can
# move past *any* 4xx/5xx the current platform returns, then try a sibling. This covers
# the openai-SDK's named subclasses (400 BadRequest, 401 Authentication, 403 Permission,
# 404 NotFound, 409 Conflict, 422 Unprocessable, 429 RateLimit, ≥500 InternalServer) AND
# the unnamed codes that some backends surface as a bare APIStatusError (e.g. 402 quota
# errors, vendor-specific permission codes). Narrower lists like (BadRequestError,
# RateLimitError, ...) miss those, which can leave a model stuck retrying one platform
# while a sibling is healthy. Stage-level retry still bounds total attempts when every
# sibling fails.
_RETRYABLE: tuple[type[Exception], ...] = (
    APIConnectionError,
    APITimeoutError,
    APIStatusError,
)


class FallbackModel(Model):
    """Tries each (label, Model) candidate in order, promoting the winner."""

    def __init__(self, candidates: list[tuple[str, Model]]) -> None:
        if not candidates:
            raise ValueError("FallbackModel requires at least one candidate")
        self._candidates = list(candidates)

    def _promote(self, index: int, label: str, failures: list[str]) -> None:
        """Move the winning candidate to the front so subsequent calls hit it first."""
        if index == 0:
            return
        logger.info(
            "fallback: served by %r after %s", label, ", ".join(failures) or "none"
        )
        self._candidates.insert(0, self._candidates.pop(index))

    async def get_response(self, *args, **kwargs) -> ModelResponse:
        """Try each candidate in order; non-retryable errors bubble immediately."""
        last_exc: Exception | None = None
        failures: list[str] = []
        for index, (label, model) in enumerate(list(self._candidates)):
            try:
                response = await model.get_response(*args, **kwargs)
                self._promote(index, label, failures)
                return response
            except _RETRYABLE as exc:
                failures.append(f"{label}({type(exc).__name__})")
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    def stream_response(self, *args, **kwargs) -> AsyncIterator[TResponseStreamEvent]:
        """Stream events; only the very first candidate boundary may switch on retryable errors."""
        return self._stream_with_fallback(*args, **kwargs)

    async def _stream_with_fallback(
        self, *args, **kwargs
    ) -> AsyncIterator[TResponseStreamEvent]:
        last_exc: Exception | None = None
        failures: list[str] = []
        for index, (label, model) in enumerate(list(self._candidates)):
            first_event_emitted = False
            try:
                async for event in model.stream_response(*args, **kwargs):
                    if not first_event_emitted:
                        self._promote(index, label, failures)
                        first_event_emitted = True
                    yield event
                return
            except _RETRYABLE as exc:
                # Mid-stream failure cannot be replayed: partial output already yielded.
                if first_event_emitted:
                    raise
                failures.append(f"{label}({type(exc).__name__})")
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    async def close(self) -> None:
        for _label, model in self._candidates:
            await model.close()
