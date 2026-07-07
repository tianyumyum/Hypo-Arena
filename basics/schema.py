"""Pydantic data contracts for the construction → generation → evaluation pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

Mode = Literal["baseline", "agent"]
Method = Literal["arena", "score"]
Pool = Literal["baseline", "agent", "full"]


REFERENCE_LABEL = "reference"


def keep_in_pool(label: str, pool: Pool) -> bool:
    """Whether a leaderboard model label belongs to the requested pool.

    Labels follow the orchestrator convention: ``baseline:<profile>``,
    ``agent:<profile>``, or the bare string ``reference``. Reference is
    always retained as a calibration anchor across pools.
    """
    if label == REFERENCE_LABEL:
        return True
    if pool == "full":
        return True
    return label.startswith(f"{pool}:")


def _utc_now() -> datetime:
    """UTC now; default factory for created_at fields."""
    return datetime.now(timezone.utc)


class HypothesisItem(BaseModel):
    """A single (hypothesis, evidence) pair, optionally categorized."""

    category: str | None = None
    evidence: str
    hypothesis: str


class SourceRecord(BaseModel):
    """One row of benchmark/<domain>/source/metadata.jsonl: a source document descriptor."""

    domain: str
    file: str | None = None                          # path relative to source/ dir
    id: str                                          # e.g., "biomedical_science:10.1016_j.celrep.2025.116174"
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = 1
    title: str = ""
    url: str | None = None                           # remote URL (e.g., postmortem fetch target or paper landing page)


# ---- construction provenance + audit ----

class TokenUsage(BaseModel):
    """Token counters mirroring agents.usage.Usage shape (per-call or accumulated)."""

    cached_tokens: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    requests: int = 0

    @computed_field
    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @classmethod
    def from_sdk_usage(cls, usage: Any) -> "TokenUsage":
        """Project an openai-agents Usage object onto our schema (duck-typed; SDK-free import)."""
        cached = getattr(getattr(usage, "input_tokens_details", None), "cached_tokens", 0) or 0
        reasoning = getattr(getattr(usage, "output_tokens_details", None), "reasoning_tokens", 0) or 0
        return cls(
            cached_tokens=cached,
            input_tokens=getattr(usage, "input_tokens", 0) or 0,
            output_tokens=getattr(usage, "output_tokens", 0) or 0,
            reasoning_tokens=reasoning,
            requests=getattr(usage, "requests", 0) or 0,
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            cached_tokens=self.cached_tokens + other.cached_tokens,
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            reasoning_tokens=self.reasoning_tokens + other.reasoning_tokens,
            requests=self.requests + other.requests,
        )


class WebSearchTrace(BaseModel):
    """One web_search tool action recorded during a Forge call."""

    action: Literal["search", "open_page", "find_in_page"]
    query: str | None = None
    url: str | None = None


class AuditIssue(BaseModel):
    """One problem flagged by an Audit pass."""

    problem: str
    revision_instruction: str
    target_pair: int | None = None
    why_it_matters: str


class AuditResult(BaseModel):
    """Outcome of one Audit pass over a Context or a set of Hypotheses."""

    issues: list[AuditIssue] = Field(default_factory=list)
    passed: bool
    summary: str


class CaseQuality(BaseModel):
    """Composite Audit verdict: passed iff both audits pass."""

    context_audit: AuditResult
    hypothesis_audit: AuditResult

    @computed_field
    @property
    def passed(self) -> bool:
        return self.context_audit.passed and self.hypothesis_audit.passed


class ConstructionProvenance(BaseModel):
    """Who/how/when produced this case."""

    context_rounds: int
    created_at: datetime = Field(default_factory=_utc_now)
    hypothesis_rounds: int
    profile: str
    tokens: TokenUsage = Field(default_factory=TokenUsage)
    web_searches: list[WebSearchTrace] = Field(default_factory=list)


# ---- aggregate: stage-1 output ----

class BenchmarkCase(BaseModel):
    """One row of benchmark/<domain>/cases/<config>.jsonl."""

    context: str
    domain: str
    hypotheses: list[HypothesisItem]              # reference (Forge output)
    id: str                                       # e.g., "biomedical_science:pmid-12345"
    metadata: dict[str, Any] = Field(default_factory=dict)
    provenance: ConstructionProvenance
    quality: CaseQuality
    schema_version: int = 1


# ---- generation provenance ----

class GenerationProvenance(BaseModel):
    """Who/how/when produced this submission."""

    created_at: datetime = Field(default_factory=_utc_now)
    fallback_platform: str | None = None
    mode: Mode
    profile: str
    skills_used: list[str] | None = None
    tokens: TokenUsage = Field(default_factory=TokenUsage)


# ---- aggregate: stage-2 output ----

class Submission(BaseModel):
    """One row of benchmark/<domain>/submissions/<mode>/<config>+<profile>.jsonl."""

    domain: str
    hypotheses: list[HypothesisItem]              # candidate (model output)
    id: str                                       # must match an existing BenchmarkCase.id
    provenance: GenerationProvenance
    schema_version: int = 1


# ---- evaluation: arena ----

class JudgeVerdict(BaseModel):
    """One direction of a pairwise comparison."""

    rationale: str = ""
    rubric_scores: dict[str, float] = Field(default_factory=dict)
    score: float                                  # A's win share, ∈ [0, 1]
    winner: Literal["a", "b", "tie"]


class ArenaMatch(BaseModel):
    """One row of arena.matches.jsonl: forward + reverse verdicts with derived stats."""

    case_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    forward: JudgeVerdict
    judge: str
    model_a: str
    model_b: str
    reverse: JudgeVerdict                         # judge sees (B, A); compute below flips it
    schema_version: int = 1

    @computed_field
    @property
    def consistent(self) -> bool:
        # Both ties agree; otherwise both must pick OPPOSITE positions
        # (forward "a" wins == A; reverse "b" wins == A — same answer).
        f, r = self.forward.winner, self.reverse.winner
        if f == "tie" or r == "tie":
            return f == r
        return f != r

    @computed_field
    @property
    def debiased_score(self) -> float:
        return (self.forward.score + (1.0 - self.reverse.score)) / 2.0


# ---- evaluation: score ----

class RecallStats(BaseModel):
    """Reference-anchored coverage stats: how many GT hypotheses the Submission hits."""

    hits: int
    total: int

    @computed_field
    @property
    def ratio(self) -> float:
        return self.hits / self.total if self.total > 0 else 0.0


class ScoreRecord(BaseModel):
    """One row of score.jsonl: per-pair + set-level dimension scores plus computed aggregates."""

    case_id: str
    created_at: datetime = Field(default_factory=_utc_now)
    judge: str
    model: str
    pair_scores: list[dict[str, float]] = Field(default_factory=list)  # one dict per Hypothesis (paper §3.1.2)
    rationale: str = ""
    recall: RecallStats | None = None                              # multi-hypothesis only; diagnostic, not in S
    schema_version: int = 1
    set_scores: dict[str, float] = Field(default_factory=dict)    # breadth/distinctness/utility (multi only)

    @computed_field
    @property
    def pair_summary(self) -> float:
        """Q_pair = mean of q_i across K submitted pairs (paper §3.1.2)."""
        q_is: list[float] = []
        for pair in self.pair_scores:
            if pair:
                q_is.append(sum(pair.values()) / len(pair))
        return sum(q_is) / len(q_is) if q_is else 0.0

    @computed_field
    @property
    def set_summary(self) -> float | None:
        if not self.set_scores:
            return None
        return sum(self.set_scores.values()) / len(self.set_scores)

    @computed_field
    @property
    def overall_score(self) -> float:
        if self.set_summary is not None:
            return (self.pair_summary + self.set_summary) / 2.0
        return self.pair_summary


# ---- evaluation: leaderboard (arena and score share this shape) ----

class LeaderboardEntry(BaseModel):
    """One ranked model in a leaderboard."""

    breakdown: dict[str, float] = Field(default_factory=dict)
    model: str
    n_observations: int
    rank: int
    rating: float                                 # arena: BTD log-scale; score: overall_score average


class LeaderboardMetadata(BaseModel):
    """Run-level metadata attached to every leaderboard."""

    btd_iterations: int | None = None
    config: str
    created_at: datetime = Field(default_factory=_utc_now)
    domain: str | None = None                     # None for cross-domain summaries
    domains: list[str] | None = None              # populated only for summaries
    judge: str
    method: Method
    n_models: int
    n_observations: int
    pool: Pool | None = None                      # arena/score pool when leaderboard is pool-filtered
    position_consistency_rate: float | None = None


class Leaderboard(BaseModel):
    """One file: <method>.leaderboard.json — metadata plus ranked entries."""

    metadata: LeaderboardMetadata
    rankings: list[LeaderboardEntry]
    schema_version: int = 1
