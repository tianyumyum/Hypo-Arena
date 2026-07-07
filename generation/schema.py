"""Generation pipeline schemas: candidate hypothesis sets + skill-selector output."""

from __future__ import annotations

from typing import Any

from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel, Field, model_validator


def _inline_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Replace all `$ref` pointers with their `$defs` targets and drop `$defs`.

    Gemini's structured-output API rejects JSON Schemas containing `$defs` / `$ref`;
    every other supported provider accepts the inlined form. Applied only to gemini
    profiles via `FlatAgentOutputSchema` below.
    """
    defs = schema.get("$defs", {})

    def resolve(node: Any, depth: int = 0) -> Any:
        if depth > 64:
            return node
        if isinstance(node, dict):
            if "$ref" in node and isinstance(node["$ref"], str) and node["$ref"].startswith("#/$defs/"):
                key = node["$ref"].split("/")[-1]
                if key in defs:
                    return resolve(defs[key], depth + 1)
            return {k: resolve(v, depth + 1) for k, v in node.items()}
        if isinstance(node, list):
            return [resolve(x, depth + 1) for x in node]
        return node

    flat = resolve(schema)
    flat.pop("$defs", None)
    return flat


class FlatAgentOutputSchema(AgentOutputSchema):
    """AgentOutputSchema variant that inlines `$defs`/`$ref` post-construction.

    Used for gemini profiles whose API rejects schemas with reference pointers.
    Behaviour is identical to AgentOutputSchema for any model whose schema is
    already flat (no `$defs`).
    """

    def __init__(self, output_type: type, strict_json_schema: bool = True) -> None:
        super().__init__(output_type, strict_json_schema=strict_json_schema)
        self._output_schema = _inline_refs(self._output_schema)


class HypothesisCandidate(BaseModel):
    """Candidate output for single-hypothesis (research) domains."""

    evidence: str
    hypothesis: str


class CategorizedHypothesisCandidate(BaseModel):
    """Multi-hypothesis candidate item: every entry carries a non-empty analytical category."""

    category: str = Field(min_length=1)
    evidence: str
    hypothesis: str


class HypothesisCandidateSet(BaseModel):
    """Candidate output for multi-hypothesis (real-world) domains."""

    hypotheses: list[CategorizedHypothesisCandidate]


class SkillSelection(BaseModel):
    """Skill selector's structured output: 1–3 skill names (Agent Mode requires at least one).

    Tolerant input shape: Claude / GLM frequently emit a bare JSON array
    `["a", "b"]` instead of the wrapped object `{"skills": ["a", "b"]}`. The validator
    below normalizes both into the same internal form. Without this, strict-JSON
    Pydantic validation rejects the bare array and the agent's structured-output call
    fails with ModelBehaviorError.
    """

    skills: list[str] = Field(min_length=1, max_length=3)

    @model_validator(mode="before")
    @classmethod
    def _accept_bare_list(cls, data):
        if isinstance(data, list):
            return {"skills": data}
        return data
