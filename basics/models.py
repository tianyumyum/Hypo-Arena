"""Model registries: Profile shape, retry policy, resolution, and per-silo profile catalogs."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from agents import ModelRetrySettings, ModelSettings, retry_policies
from agents.models.interface import Model
from openai.types.shared import Reasoning

from .platform import Platform, get_client, is_available

logger = logging.getLogger("hypo.agent")


# ---- shared retry policy ----

_POLICY = retry_policies.any(
    retry_policies.provider_suggested(),
    retry_policies.retry_after(),
    retry_policies.network_error(),
    retry_policies.http_status([408, 409, 429, 500, 502, 503, 504]),
)

DEFAULT_RETRY = ModelRetrySettings(
    max_retries=3,
    backoff={
        "initial_delay": 1.0,
        "max_delay": 30.0,
        "multiplier": 2.0,
        "jitter": True,
    },
    policy=_POLICY,
)


# ---- profile data shape + resolution ----

@dataclass(frozen=True)
class Profile:
    """A logical model profile and its per-platform fallback chain."""

    model_settings: ModelSettings
    platform_ids: list[tuple[Platform, str | None]]
    # False for backends that reject `response_format: json_schema` at the wire level
    # (e.g. Doubao seed-2.0-pro). Callers needing structured output should fall back
    # to plain-text + manual JSON parse for these profiles.
    supports_response_format: bool = True


_warned_skips: set[tuple[str, str, str]] = set()


def _warn_once(profile_name: str, label: str, reason: str) -> None:
    """Log a one-shot warning per (profile, label, reason) triple."""
    key = (profile_name, label, reason)
    if key in _warned_skips:
        return
    _warned_skips.add(key)
    logger.warning(
        "Profile %r: skipping platform %r (%s)", profile_name, label, reason
    )


def resolve_candidates(
    profile_name: str,
    profile: Profile,
    model_class: type[Model],
) -> list[tuple[str, Model]]:
    """Build the FallbackModel candidates for a profile, skipping unconfigured platforms."""
    candidates: list[tuple[str, Model]] = []
    for label, model_id in profile.platform_ids:
        if model_id is None:
            _warn_once(profile_name, label, "model_id not configured")
            continue
        if not is_available(label):
            _warn_once(profile_name, label, "env vars missing")
            continue
        candidates.append(
            (label, model_class(model=model_id, openai_client=get_client(label)))
        )
    if not candidates:
        configured = [label for label, _ in profile.platform_ids]
        raise RuntimeError(
            f"No usable platforms for profile {profile_name!r}. "
            f"Tried: {configured}; check env vars and replace placeholder model IDs."
        )
    return candidates


# ---- profile builders (one per API form) ----

_TYPED_REASONING_EFFORTS = {"minimal", "low", "medium", "high"}


def _responses_profile(
    platforms: list[tuple[Platform, str]], *, effort: str,
) -> Profile:
    """Responses-API Profile: typed Reasoning for standard efforts, extra_body otherwise."""
    if effort in _TYPED_REASONING_EFFORTS:
        settings = ModelSettings(
            max_tokens=128000,
            reasoning=Reasoning(effort=effort),
            retry=DEFAULT_RETRY,
        )
    else:
        settings = ModelSettings(
            max_tokens=128000,
            extra_body={"reasoning": {"effort": effort}},
            retry=DEFAULT_RETRY,
        )
    return Profile(platform_ids=platforms, model_settings=settings)


def _chat_profile(
    platforms: list[tuple[Platform, str]],
    *,
    supports_response_format: bool = True,
    **extra_body: Any,
) -> Profile:
    """Chat-Completions Profile; extra_body keys vary per vendor."""
    return Profile(
        platform_ids=platforms,
        model_settings=ModelSettings(
            max_tokens=65536,
            extra_body=extra_body or None,
            retry=DEFAULT_RETRY,
        ),
        supports_response_format=supports_response_format,
    )


# ---- platform-id maps (one entry per model family) ----
_CLAUDE_SONNET_4_6 = [
    ("platform_a",   "claude-sonnet-4-6-20260217"),
    ("platform_d", "claude-sonnet-4-6-20260217"),
    ("platform_c", "claude-sonnet-4-6"),
]

_CLAUDE_OPUS_4_8 = [
    ("platform_d", "claude-opus-4-8"),
]
_CLAUDE_OPUS_4_7 = [
    ("platform_d", "claude-opus-4-7"),
]

_CLAUDE_OPUS_4_6 = [
    ("platform_a",   "claude-opus-4-6-20260205"),
    ("platform_d", "claude-opus-4-6-20260205"),
    ("platform_c", "claude-opus-4-6"),
]
_DEEPSEEK_V4_FLASH = [
    ("platform_b", "deepseek-v4-flash"),
    ("platform_d", "deepseek-v4-flash"),
]
_DEEPSEEK_V4_PRO   = [
    ("platform_b", "deepseek-v4-pro"),
    ("platform_d", "deepseek-v4-pro"),
]
_GEMINI_3_FLASH = [
    ("platform_a",   "gemini-3-flash-preview"),
    ("platform_d", "gemini-3-flash-preview"),
    ("platform_c", "gemini-3-flash-preview"),
]

_GEMINI_3_5_FLASH = [
    ("platform_d",   "gemini-3.5-flash"),
]
_GEMINI_3_1_PRO = [
    ("platform_a",   "gemini-3.1-pro-preview"),
    ("platform_d", "gemini-3.1-pro-preview"),
    ("platform_c", "gemini-3.1-pro-preview"),
]
_GLM_5 = [
    ("platform_a",   "glm-5"),
    ("platform_b", "glm-5"),
    ("platform_d", "glm-5"),
    ("platform_c", "platform_b/glm-5"),
]
_GLM_5_1 = [
    ("platform_a",   "glm-5.1"),
    ("platform_b", "glm-5.1"),
    ("platform_d", "glm-5.1"),
    ("platform_c", "platform_b/glm-5.1"),
]
_GLM_5_2 = [
    ("platform_d", "glm-5.2"),
]
_GPT_5_4_MINI = [
    ("platform_c", "gpt-5.4-mini-0317-global"),
]
_GPT_5_4 = [
    ("platform_a",   "gpt-5.4-2026-03-05"),
    ("platform_d", "gpt-5.4-2026-03-05"),
    ("platform_c", "gpt-5.4-0305-global"),
]
_GPT_5_5 = [
    ("platform_d", "gpt-5.5"),
    ("platform_c", "gpt-5.5-0424-global"),
]
_KIMI_K2_5 = [
    ("platform_a",   "kimi-k2.5"),
    ("platform_b", "kimi-k2.5"),
    ("platform_d", "kimi-k2.5"),
    ("platform_c", "platform_b/kimi-k2.5"),
]
_KIMI_K2_6 = [
    ("platform_a",   "moonshot/kimi-k2.6"),
    ("platform_b", "kimi-k2.6"),
    ("platform_d", "kimi-k2.6"),
    ("platform_c", "moonshot/kimi-k2.6"),
]
_MIMO_V2_FLASH = [
    ("platform_a",   "mimo-v2-flash"),
    ("platform_d", "mimo-v2-flash"),
]
_MIMO_V2_PRO = [
    # ("platform_a",   "mimo-v2-pro"),
    ("platform_d", "MiMo-V2-Pro"),
]
_MINIMAX_M2_5 = [
    ("platform_a",   "MiniMax-M2.5"),
    ("platform_b", "MiniMax-M2.5"),
    ("platform_d", "MiniMax-M2.5"),
    ("platform_c", "MiniMax/MiniMax-M2.5"),
]
_MINIMAX_M2_7 = [
    ("platform_a",   "MiniMax-M2.7"),
    ("platform_b", "MiniMax-M2.7"),
    ("platform_d", "MiniMax-M2.7"),
    ("platform_c", "MiniMax/MiniMax-M2.7"),
]
_QWEN_3_6_MAX = [
    ("platform_b", "qwen3.6-max-preview"),
    ("platform_c", "qwen3.6-max-preview"),
]
_QWEN_3_7_MAX = [
    ("platform_b", "qwen3.7-max"),
    ("platform_c", "qwen3.7-max"),
]
_SEED_2_0_PRO = [
    ("platform_a",   "doubao-seed-2-0-pro-260215"),
    ("platform_d", "doubao-seed-2-0-pro-260215"),
]

_SEED_2_1_PRO = [
    ("platform_d", "doubao-seed-2-1-pro-260628"),
]


_MIMO_V2_5_PRO = [
    ("platform_d", "MiMo-V2.5-Pro"),
]

_MINIMAX_M3 = [
    ("platform_d", "MiniMax-M3"),
]

# ---- construction silo: Responses API ----

CONSTRUCTION_REGISTRY: dict[str, Profile] = {
    "gpt-5.4":        _responses_profile(_GPT_5_4, effort="none"),
    "gpt-5.4-low":    _responses_profile(_GPT_5_4, effort="low"),
    "gpt-5.4-medium": _responses_profile(_GPT_5_4, effort="medium"),
    "gpt-5.4-high":   _responses_profile(_GPT_5_4, effort="high"),
    "gpt-5.4-xhigh":  _responses_profile(_GPT_5_4, effort="xhigh"),
    "gpt-5.5":        _responses_profile(_GPT_5_5, effort="none"),
    "gpt-5.5-low":    _responses_profile(_GPT_5_5, effort="low"),
    "gpt-5.5-medium": _responses_profile(_GPT_5_5, effort="medium"),
    "gpt-5.5-high":   _responses_profile(_GPT_5_5, effort="high"),
    "gpt-5.5-xhigh":  _responses_profile(_GPT_5_5, effort="xhigh"),
}


# ---- generation/eval silo: Chat Completions API ----

GENERATION_REGISTRY: dict[str, Profile] = {
    # Claude Sonnet 4.6
    "claude-sonnet-4.6":         _chat_profile(_CLAUDE_SONNET_4_6, thinking={"type": "disabled"}),
    "claude-sonnet-4.6-low":     _chat_profile(_CLAUDE_SONNET_4_6, thinking={"type": "adaptive"}, output_config={"effort": "low"}),
    "claude-sonnet-4.6-medium":  _chat_profile(_CLAUDE_SONNET_4_6, thinking={"type": "adaptive"}, output_config={"effort": "medium"}),
    "claude-sonnet-4.6-high":    _chat_profile(_CLAUDE_SONNET_4_6, thinking={"type": "adaptive"}, output_config={"effort": "high"}),
    "claude-sonnet-4.6-max":     _chat_profile(_CLAUDE_SONNET_4_6, thinking={"type": "adaptive"}, output_config={"effort": "max"}),
    # Claude Opus 4.6
    "claude-opus-4.6":           _chat_profile(_CLAUDE_OPUS_4_6,   thinking={"type": "disabled"}),
    "claude-opus-4.6-low":       _chat_profile(_CLAUDE_OPUS_4_6,   thinking={"type": "adaptive"}, output_config={"effort": "low"}),
    "claude-opus-4.6-medium":    _chat_profile(_CLAUDE_OPUS_4_6,   thinking={"type": "adaptive"}, output_config={"effort": "medium"}),
    "claude-opus-4.6-high":      _chat_profile(_CLAUDE_OPUS_4_6,   thinking={"type": "adaptive"}, output_config={"effort": "high"}),
    "claude-opus-4.6-max":       _chat_profile(_CLAUDE_OPUS_4_6,   thinking={"type": "adaptive"}, output_config={"effort": "max"}),
    # Claude Opus 4.8
    "claude-opus-4.8":           _chat_profile(_CLAUDE_OPUS_4_8,   thinking={"type": "disabled"}),
    "claude-opus-4.8-low":       _chat_profile(_CLAUDE_OPUS_4_8,   thinking={"type": "adaptive"}, output_config={"effort": "low"}),
    "claude-opus-4.8-medium":    _chat_profile(_CLAUDE_OPUS_4_8,   thinking={"type": "adaptive"}, output_config={"effort": "medium"}),
    "claude-opus-4.8-high":      _chat_profile(_CLAUDE_OPUS_4_8,   thinking={"type": "adaptive"}, output_config={"effort": "high"}),
    "claude-opus-4.8-max":       _chat_profile(_CLAUDE_OPUS_4_8,   thinking={"type": "adaptive"}, output_config={"effort": "max"}),
    # Claude Opus 4.7
    "claude-opus-4.7":           _chat_profile(_CLAUDE_OPUS_4_7,   thinking={"type": "disabled"}),
    "claude-opus-4.7-low":       _chat_profile(_CLAUDE_OPUS_4_7,   thinking={"type": "adaptive"}, output_config={"effort": "low"}),
    "claude-opus-4.7-medium":    _chat_profile(_CLAUDE_OPUS_4_7,   thinking={"type": "adaptive"}, output_config={"effort": "medium"}),
    "claude-opus-4.7-high":      _chat_profile(_CLAUDE_OPUS_4_7,   thinking={"type": "adaptive"}, output_config={"effort": "high"}),
    "claude-opus-4.7-max":       _chat_profile(_CLAUDE_OPUS_4_7,   thinking={"type": "adaptive"}, output_config={"effort": "max"}),
    # DeepSeek v4
    "deepseek-v4-flash":         _chat_profile(_DEEPSEEK_V4_FLASH, thinking={"type": "disabled"}),
    "deepseek-v4-flash-high":    _chat_profile(_DEEPSEEK_V4_FLASH, thinking={"type": "enabled"}, reasoning_effort="high"),
    "deepseek-v4-flash-max":     _chat_profile(_DEEPSEEK_V4_FLASH, thinking={"type": "enabled"}, reasoning_effort="max"),
    "deepseek-v4-pro":           _chat_profile(_DEEPSEEK_V4_PRO,   thinking={"type": "disabled"}),
    "deepseek-v4-pro-high":      _chat_profile(_DEEPSEEK_V4_PRO,   thinking={"type": "enabled"}, reasoning_effort="high"),
    "deepseek-v4-pro-max":       _chat_profile(_DEEPSEEK_V4_PRO,   thinking={"type": "enabled"}, reasoning_effort="max"),
    # Gemini 3
    "gemini-3-flash-minimal":    _chat_profile(_GEMINI_3_FLASH, reasoning_effort="minimal"),
    "gemini-3-flash-low":        _chat_profile(_GEMINI_3_FLASH, reasoning_effort="low"),
    "gemini-3-flash-medium":     _chat_profile(_GEMINI_3_FLASH, reasoning_effort="medium"),
    "gemini-3-flash-high":       _chat_profile(_GEMINI_3_FLASH, reasoning_effort="high"),
    "gemini-3.1-pro-low":        _chat_profile(_GEMINI_3_1_PRO, reasoning_effort="low"),
    "gemini-3.1-pro-medium":     _chat_profile(_GEMINI_3_1_PRO, reasoning_effort="medium"),
    "gemini-3.1-pro-high":       _chat_profile(_GEMINI_3_1_PRO, reasoning_effort="high"),
    # Gemini 3.5 Flash
    "gemini-3.5-flash-minimal":  _chat_profile(_GEMINI_3_5_FLASH, reasoning_effort="minimal"),
    "gemini-3.5-flash-low":      _chat_profile(_GEMINI_3_5_FLASH, reasoning_effort="low"),
    "gemini-3.5-flash-medium":   _chat_profile(_GEMINI_3_5_FLASH, reasoning_effort="medium"),
    "gemini-3.5-flash-high":     _chat_profile(_GEMINI_3_5_FLASH, reasoning_effort="high"),
    # GLM 5 / 5.1
    # GLM tends to wrap structured output in markdown code fences (```json ... ```);
    # SDK strict json.loads chokes. Use plain-text + manual extract_json instead.
    "glm-5":                     _chat_profile(_GLM_5,   supports_response_format=False, thinking={"type": "disabled"}),
    "glm-5-thinking":            _chat_profile(_GLM_5,   supports_response_format=False, thinking={"type": "enabled"}),
    "glm-5.1":                   _chat_profile(_GLM_5_1, supports_response_format=False, thinking={"type": "disabled"}),
    "glm-5.1-thinking":          _chat_profile(_GLM_5_1, supports_response_format=False, thinking={"type": "enabled"}),
    "glm-5.2":                    _chat_profile(_GLM_5_2, supports_response_format=False, thinking={"type": "disabled"}),
    "glm-5.2-thinking":           _chat_profile(_GLM_5_2, supports_response_format=False, thinking={"type": "enabled"}),
    # GPT 5.4 (chat completions; same vendor as construction silo, different API form)
    "gpt-5.4-mini":              _chat_profile(_GPT_5_4_MINI, reasoning_effort="none"),
    "gpt-5.4-mini-low":          _chat_profile(_GPT_5_4_MINI, reasoning_effort="low"),
    "gpt-5.4-mini-medium":       _chat_profile(_GPT_5_4_MINI, reasoning_effort="medium"),
    "gpt-5.4-mini-high":         _chat_profile(_GPT_5_4_MINI, reasoning_effort="high"),
    "gpt-5.4-mini-xhigh":        _chat_profile(_GPT_5_4_MINI, reasoning_effort="xhigh"),
    "gpt-5.4":                   _chat_profile(_GPT_5_4, reasoning_effort="none"),
    "gpt-5.4-low":               _chat_profile(_GPT_5_4, reasoning_effort="low"),
    "gpt-5.4-medium":            _chat_profile(_GPT_5_4, reasoning_effort="medium"),
    "gpt-5.4-high":              _chat_profile(_GPT_5_4, reasoning_effort="high"),
    "gpt-5.4-xhigh":             _chat_profile(_GPT_5_4, reasoning_effort="xhigh"),
    # GPT 5.5
    "gpt-5.5":                   _chat_profile(_GPT_5_5, reasoning_effort="none"),
    "gpt-5.5-low":               _chat_profile(_GPT_5_5, reasoning_effort="low"),
    "gpt-5.5-medium":            _chat_profile(_GPT_5_5, reasoning_effort="medium"),
    "gpt-5.5-high":              _chat_profile(_GPT_5_5, reasoning_effort="high"),
    "gpt-5.5-xhigh":             _chat_profile(_GPT_5_5, reasoning_effort="xhigh"),
    # Kimi K2.5 / K2.6
    "kimi-k2.5":                 _chat_profile(_KIMI_K2_5, thinking={"type": "disabled"}),
    "kimi-k2.5-thinking":        _chat_profile(_KIMI_K2_5, thinking={"type": "enabled"}),
    "kimi-k2.6":                 _chat_profile(_KIMI_K2_6, thinking={"type": "disabled"}),
    "kimi-k2.6-thinking":        _chat_profile(_KIMI_K2_6, thinking={"type": "enabled"}),
    # MiniMax M2.5 / M2.7
    "minimax-m2.5-thinking":     _chat_profile(_MINIMAX_M2_5, reasoning_split=True),
    "minimax-m2.7-thinking":     _chat_profile(_MINIMAX_M2_7, reasoning_split=True),
    "minimax-m3-thinking":       _chat_profile(_MINIMAX_M3, reasoning_split=True),
    # Qwen 3.6 Plus / Max
    "qwen-3.6-max":              _chat_profile(_QWEN_3_6_MAX, enable_thinking=False),
    "qwen-3.6-max-thinking":     _chat_profile(_QWEN_3_6_MAX, enable_thinking=True),
    "qwen-3.7-max":              _chat_profile(_QWEN_3_7_MAX, enable_thinking=False),
    "qwen-3.7-max-thinking":     _chat_profile(_QWEN_3_7_MAX, enable_thinking=True),
    # Judges
    "mimo-v2-flash":             _chat_profile(_MIMO_V2_FLASH, thinking={"type": "disabled"}),
    "mimo-v2-pro":               _chat_profile(_MIMO_V2_PRO, thinking={"type": "disabled"}),
    # mimo-v2.5-pro accepts response_format but ignores the schema without a prompt-side
    # directive (returns arbitrary keys); the text + extract_json path tested 5/5, so pin it.
    "mimo-v2.5-pro":             _chat_profile(_MIMO_V2_5_PRO, supports_response_format=False, thinking={"type": "disabled"}),
    # Seed: backend rejects response_format=json_schema (verified via probe);
    # judge agents must use plain text + manual JSON parsing.
    "seed-2.0-pro":              _chat_profile(_SEED_2_0_PRO, supports_response_format=False, thinking={"type": "disabled"}),
    "seed-2.0-pro-minimal":      _chat_profile(_SEED_2_0_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="minimal"),
    "seed-2.0-pro-low":          _chat_profile(_SEED_2_0_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="low"),
    "seed-2.0-pro-medium":       _chat_profile(_SEED_2_0_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="medium"),
    "seed-2.0-pro-high":         _chat_profile(_SEED_2_0_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="high"),
    # Seed 2.1 Pro: assume same json_schema limitation as 2.0 (verify via probe_platform).
    "seed-2.1-pro":              _chat_profile(_SEED_2_1_PRO, supports_response_format=False, thinking={"type": "disabled"}),
    "seed-2.1-pro-minimal":      _chat_profile(_SEED_2_1_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="minimal"),
    "seed-2.1-pro-low":          _chat_profile(_SEED_2_1_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="low"),
    "seed-2.1-pro-medium":       _chat_profile(_SEED_2_1_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="medium"),
    "seed-2.1-pro-high":         _chat_profile(_SEED_2_1_PRO, supports_response_format=False, thinking={"type": "enabled"}, reasoning_effort="high"),
}
