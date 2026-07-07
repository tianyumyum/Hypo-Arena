"""Defensive JSON-text parser used by judges that can't speak response_format=json_schema."""

from __future__ import annotations

import json

import pytest

from basics.parsing import coerce_to_model, extract_json
from evaluation.schema import ArenaVerdict, ScoreVerdict


def test_extract_json_clean():
    text = '{"verdict": "A>B", "rationale": "concise"}'
    assert extract_json(text) == {"verdict": "A>B", "rationale": "concise"}


def test_extract_json_strips_markdown_fence():
    text = '```json\n{"verdict": "A=B", "rationale": "tied"}\n```'
    assert extract_json(text) == {"verdict": "A=B", "rationale": "tied"}


def test_extract_json_strips_bare_fence():
    text = '```\n{"verdict": "B>>A", "rationale": "decisive"}\n```'
    assert extract_json(text) == {"verdict": "B>>A", "rationale": "decisive"}


def test_extract_json_with_prose_around():
    text = "Here is my verdict:\n{\"verdict\": \"A>B\", \"rationale\": \"better\"}\nHope this helps."
    assert extract_json(text) == {"verdict": "A>B", "rationale": "better"}


def test_extract_json_with_trailing_comma():
    text = '{"verdict": "A=B", "rationale": "ok",}'
    assert extract_json(text) == {"verdict": "A=B", "rationale": "ok"}


def test_extract_json_with_invalid_escape():
    # \p is not a valid JSON escape; parser should fix by escaping the backslash
    text = r'{"verdict": "A>B", "rationale": "see \page 5"}'
    out = extract_json(text)
    assert out["verdict"] == "A>B"
    assert "page 5" in out["rationale"]


def test_extract_json_with_literal_newline_in_string():
    # Literal newline mid-string would break json.loads; parser should escape it
    text = '{"verdict": "A>B", "rationale": "first line\nsecond line"}'
    out = extract_json(text)
    assert out["verdict"] == "A>B"
    assert "first line" in out["rationale"] and "second line" in out["rationale"]


def test_coerce_passthrough_for_already_validated_model():
    v = ArenaVerdict(verdict="A>B", rationale="x")
    assert coerce_to_model(v, ArenaVerdict) is v


def test_coerce_text_to_arena_verdict():
    text = '{"verdict": "A>>B", "rationale": "A is decisively grounded"}'
    out = coerce_to_model(text, ArenaVerdict)
    assert out.verdict == "A>>B"
    assert out.rationale == "A is decisively grounded"


def test_coerce_text_to_score_verdict_full():
    text = json.dumps({
        "pair_scores": [
            {"grounding": 4, "insight": 3, "justification": 4},
            {"grounding": 3, "insight": 4, "justification": 3},
        ],
        "set_scores": {"breadth": 4, "distinctness": 5, "utility": 3},
        "rationale": "balanced",
        "recall": "2/2",
    })
    out = coerce_to_model(text, ScoreVerdict)
    assert len(out.pair_scores) == 2
    assert out.pair_scores[0].grounding == 4
    assert out.set_scores.distinctness == 5
    assert out.recall == "2/2"


def test_coerce_text_to_score_verdict_with_nulls():
    text = json.dumps({
        "pair_scores": [{"grounding": 5, "insight": 5, "justification": 5}],
        "set_scores": None,
        "rationale": "single",
        "recall": None,
    })
    out = coerce_to_model(text, ScoreVerdict)
    assert out.set_scores is None
    assert out.recall is None


def test_coerce_invalid_input_raises():
    with pytest.raises(TypeError):
        coerce_to_model(42, ArenaVerdict)
