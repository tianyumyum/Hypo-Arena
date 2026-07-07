"""Sanity checks for CONSTRUCTION_REGISTRY and GENERATION_REGISTRY entries."""

from __future__ import annotations

import pytest

from basics.models import CONSTRUCTION_REGISTRY, GENERATION_REGISTRY

_VALID_PLATFORMS = {"platform_a", "platform_b", "platform_c", "platform_d"}


# ---- construction registry ----

def test_construction_covers_gpt_5_4_and_5_5_families():
    expected = {
        "gpt-5.4", "gpt-5.4-low", "gpt-5.4-medium", "gpt-5.4-high", "gpt-5.4-xhigh",
        "gpt-5.5", "gpt-5.5-low", "gpt-5.5-medium", "gpt-5.5-high", "gpt-5.5-xhigh",
    }
    assert set(CONSTRUCTION_REGISTRY) == expected


def test_construction_platforms_are_responses_capable():
    """Construction uses only Responses-API-capable platforms; platform_b is Chat-only."""
    responses_capable = {"platform_a", "platform_c", "platform_d"}
    for name, profile in CONSTRUCTION_REGISTRY.items():
        labels = {label for label, _ in profile.platform_ids}
        assert labels <= responses_capable, f"{name}: non-Responses platform in {labels}"
        assert labels, f"{name}: no platforms configured"


def test_construction_max_tokens_128k():
    for name, profile in CONSTRUCTION_REGISTRY.items():
        assert profile.model_settings.max_tokens == 128000, name


# ---- generation registry ----

def test_generation_registry_has_expected_families():
    families = {
        "claude-sonnet-4.6", "claude-opus-4.6",
        "deepseek-v4-flash", "deepseek-v4-pro",
        "gemini-3-flash", "gemini-3.1-pro",
        "glm-5", "glm-5.1",
        "gpt-5.4-mini", "gpt-5.4",
        "kimi-k2.5", "kimi-k2.6",
        "minimax-m2.5", "minimax-m2.7",
        "qwen-3.6-plus", "qwen-3.6-max",
        "grok-4.1-fast", "grok-4.1-fast-reasoning",
        "seed-2.0-pro",
    }
    seen = {name.split("-thinking")[0].rsplit("-", 1)[0] if "-" in name else name
            for name in GENERATION_REGISTRY}
    # Loose check: every family base name appears as a prefix of at least one registered profile.
    for fam in families:
        assert any(p == fam or p.startswith(fam + "-") for p in GENERATION_REGISTRY), fam


def test_generation_registry_size_matches_hypoarena_extent():
    # 47 inference profiles + 7 judge profiles (seed-2.0-pro, grok-4.1-*) + 2 mimo judge candidates = 56
    assert len(GENERATION_REGISTRY) == 56


def test_judges_are_present():
    for judge in ("grok-4.1-fast", "seed-2.0-pro"):
        assert judge in GENERATION_REGISTRY


def test_generation_max_tokens_65k():
    for name, profile in GENERATION_REGISTRY.items():
        assert profile.model_settings.max_tokens == 65536, name


def test_all_profile_platform_labels_are_valid():
    for registry in (CONSTRUCTION_REGISTRY, GENERATION_REGISTRY):
        for name, profile in registry.items():
            for label, _model_id in profile.platform_ids:
                assert label in _VALID_PLATFORMS, f"{name}: bad label {label!r}"


def test_no_profile_has_none_model_id():
    """Every registered profile must have a real model_id on at least one platform."""
    for registry in (CONSTRUCTION_REGISTRY, GENERATION_REGISTRY):
        for name, profile in registry.items():
            assert any(mid is not None for _, mid in profile.platform_ids), (
                f"{name}: all platform model_ids are None"
            )


@pytest.mark.parametrize("registry", [CONSTRUCTION_REGISTRY, GENERATION_REGISTRY])
def test_every_profile_has_default_retry(registry):
    for name, profile in registry.items():
        assert profile.model_settings.retry is not None, name


# ---- spot checks for reasoning fields ----

def test_construction_high_uses_typed_reasoning():
    """gpt-5.4-high uses typed Reasoning(effort='high'), not extra_body."""
    settings = CONSTRUCTION_REGISTRY["gpt-5.4-high"].model_settings
    assert settings.reasoning is not None
    assert settings.reasoning.effort == "high"
    assert settings.extra_body is None


def test_construction_xhigh_uses_extra_body():
    """gpt-5.4-xhigh uses extra_body since 'xhigh' is not in SDK Reasoning enum."""
    settings = CONSTRUCTION_REGISTRY["gpt-5.4-xhigh"].model_settings
    assert settings.reasoning is None
    assert settings.extra_body == {"reasoning": {"effort": "xhigh"}}


def test_generation_claude_high_carries_thinking_and_effort():
    settings = GENERATION_REGISTRY["claude-opus-4.6-high"].model_settings
    assert settings.extra_body == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }


def test_generation_deepseek_high_carries_thinking_and_reasoning_effort():
    settings = GENERATION_REGISTRY["deepseek-v4-pro-high"].model_settings
    assert settings.extra_body == {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    }


def test_generation_qwen_uses_enable_thinking_flag():
    on = GENERATION_REGISTRY["qwen-3.6-plus-thinking"].model_settings
    off = GENERATION_REGISTRY["qwen-3.6-plus"].model_settings
    assert on.extra_body == {"enable_thinking": True}
    assert off.extra_body == {"enable_thinking": False}
