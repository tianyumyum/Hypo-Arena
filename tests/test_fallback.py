"""Unit tests for FallbackModel — no network, all errors injected."""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest
from openai import (
    APIConnectionError,
    BadRequestError,
    InternalServerError,
    RateLimitError,
)

from agents.items import ModelResponse
from agents.usage import Usage

from basics.platform import FallbackModel

_FAKE_REQUEST = httpx.Request("POST", "https://test.local/v1/chat/completions")


def _ok_response() -> ModelResponse:
    """A minimal ModelResponse that satisfies the SDK's dataclass shape."""
    return ModelResponse(output=[], usage=Usage(requests=1), response_id=None)


def _connection_err() -> APIConnectionError:
    return APIConnectionError(request=_FAKE_REQUEST)


def _rate_limit_err() -> RateLimitError:
    return RateLimitError(
        "rate limited", response=httpx.Response(429, request=_FAKE_REQUEST), body=None
    )


def _server_err() -> InternalServerError:
    return InternalServerError(
        "oops", response=httpx.Response(500, request=_FAKE_REQUEST), body=None
    )


def _bad_request_err() -> BadRequestError:
    return BadRequestError(
        "bad", response=httpx.Response(400, request=_FAKE_REQUEST), body=None
    )


class FakeModel:
    """Records calls; raises configured exception or returns ok_response."""

    def __init__(self, raises: Exception | None = None) -> None:
        self.raises = raises
        self.calls = 0

    async def get_response(self, *args, **kwargs):
        self.calls += 1
        if self.raises is not None:
            raise self.raises
        return _ok_response()

    def stream_response(self, *args, **kwargs):
        raise NotImplementedError

    async def close(self) -> None:
        pass


class FakeStreamModel:
    """Yields N events, optionally raising before/after some events."""

    def __init__(
        self,
        *,
        events_to_yield: int = 3,
        raise_after: int | None = None,
        raises: Exception | None = None,
    ) -> None:
        self.events_to_yield = events_to_yield
        self.raise_after = raise_after
        self.raises = raises
        self.calls = 0

    def stream_response(self, *args, **kwargs) -> AsyncIterator:
        self.calls += 1
        return self._gen()

    async def _gen(self):
        emitted = 0
        if self.raises is not None and self.raise_after == 0:
            raise self.raises
        for i in range(self.events_to_yield):
            yield {"type": "delta", "index": i}
            emitted += 1
            if self.raises is not None and emitted == self.raise_after:
                raise self.raises

    async def get_response(self, *args, **kwargs):
        raise NotImplementedError

    async def close(self) -> None:
        pass


def _candidate_labels(fb: FallbackModel) -> list[str]:
    """Test-only access to the internal candidate order."""
    return [label for label, _ in fb._candidates]


# ---- get_response ----

async def test_first_candidate_succeeds_no_promotion():
    a, b = FakeModel(), FakeModel()
    fb = FallbackModel([("a", a), ("b", b)])

    result = await fb.get_response()

    assert isinstance(result, ModelResponse)
    assert (a.calls, b.calls) == (1, 0)
    assert _candidate_labels(fb) == ["a", "b"]


async def test_retryable_falls_through_and_promotes():
    a = FakeModel(raises=_rate_limit_err())
    b = FakeModel()
    c = FakeModel()
    fb = FallbackModel([("a", a), ("b", b), ("c", c)])

    result = await fb.get_response()

    assert isinstance(result, ModelResponse)
    assert (a.calls, b.calls, c.calls) == (1, 1, 0)
    assert _candidate_labels(fb) == ["b", "a", "c"]


async def test_all_retryable_raises_last_exception():
    a = FakeModel(raises=_connection_err())
    b = FakeModel(raises=_rate_limit_err())
    c = FakeModel(raises=_server_err())
    fb = FallbackModel([("a", a), ("b", b), ("c", c)])

    with pytest.raises(InternalServerError):
        await fb.get_response()

    assert (a.calls, b.calls, c.calls) == (1, 1, 1)
    assert _candidate_labels(fb) == ["a", "b", "c"]


async def test_bad_request_falls_through_to_sibling_platform():
    # Vendor-specific 400s (quota, routing) surface as BadRequestError; treat as retryable
    # so a sibling platform can still serve the request.
    a = FakeModel(raises=_bad_request_err())
    b = FakeModel()
    fb = FallbackModel([("a", a), ("b", b)])

    result = await fb.get_response()

    assert isinstance(result, ModelResponse)
    assert (a.calls, b.calls) == (1, 1)
    assert _candidate_labels(fb) == ["b", "a"]


async def test_empty_candidates_rejected():
    with pytest.raises(ValueError, match="at least one candidate"):
        FallbackModel([])


# ---- stream_response ----

async def _drain(stream: AsyncIterator) -> list:
    return [event async for event in stream]


async def test_stream_first_candidate_succeeds():
    a = FakeStreamModel(events_to_yield=3)
    b = FakeStreamModel(events_to_yield=3)
    fb = FallbackModel([("a", a), ("b", b)])

    events = await _drain(fb.stream_response())

    assert len(events) == 3
    assert (a.calls, b.calls) == (1, 0)
    assert _candidate_labels(fb) == ["a", "b"]


async def test_stream_pre_yield_failure_falls_through_and_promotes():
    a = FakeStreamModel(events_to_yield=2, raise_after=0, raises=_rate_limit_err())
    b = FakeStreamModel(events_to_yield=2)
    fb = FallbackModel([("a", a), ("b", b)])

    events = await _drain(fb.stream_response())

    assert len(events) == 2
    assert (a.calls, b.calls) == (1, 1)
    assert _candidate_labels(fb) == ["b", "a"]


async def test_stream_mid_stream_failure_does_not_fallback():
    a = FakeStreamModel(events_to_yield=3, raise_after=1, raises=_rate_limit_err())
    b = FakeStreamModel(events_to_yield=3)
    fb = FallbackModel([("a", a), ("b", b)])

    collected: list = []
    with pytest.raises(RateLimitError):
        async for event in fb.stream_response():
            collected.append(event)

    assert len(collected) == 1
    assert (a.calls, b.calls) == (1, 0)
    assert _candidate_labels(fb) == ["a", "b"]


async def test_stream_all_fail_pre_yield_raises_last():
    a = FakeStreamModel(events_to_yield=2, raise_after=0, raises=_connection_err())
    b = FakeStreamModel(events_to_yield=2, raise_after=0, raises=_server_err())
    fb = FallbackModel([("a", a), ("b", b)])

    with pytest.raises(InternalServerError):
        await _drain(fb.stream_response())

    assert (a.calls, b.calls) == (1, 1)
    assert _candidate_labels(fb) == ["a", "b"]
