"""Foundation layer: data contracts + LLM transport shared by every pipeline."""

from . import io
from .chat import build_chat_agent
from .domain import (
    ALL_DOMAINS,
    DOMAINS,
    REAL_WORLD_DOMAINS,
    RESEARCH_DOMAINS,
    DomainConfig,
    get_domain,
)
from .models import CONSTRUCTION_REGISTRY, GENERATION_REGISTRY
from .paths import (
    ARTIFACTS_ROOT,
    arena_leaderboard_path,
    arena_matches_path,
    arena_pool_leaderboard_path,
    cases_path,
    score_leaderboard_path,
    score_pool_leaderboard_path,
    score_records_path,
    source_dir,
    submission_path,
    submissions_glob,
    summary_path,
)
from .responses import build_responses_agent
from .runtime import configure_runtime
from .schema import (
    ArenaMatch,
    AuditIssue,
    AuditResult,
    BenchmarkCase,
    CaseQuality,
    ConstructionProvenance,
    GenerationProvenance,
    HypothesisItem,
    JudgeVerdict,
    Leaderboard,
    LeaderboardEntry,
    LeaderboardMetadata,
    Method,
    Mode,
    Pool,
    REFERENCE_LABEL,
    RecallStats,
    ScoreRecord,
    SourceRecord,
    Submission,
    TokenUsage,
    WebSearchTrace,
    keep_in_pool,
)

__all__ = [
    # schema
    "ArenaMatch",
    "AuditIssue",
    "AuditResult",
    "BenchmarkCase",
    "CaseQuality",
    "ConstructionProvenance",
    "GenerationProvenance",
    "HypothesisItem",
    "JudgeVerdict",
    "Leaderboard",
    "LeaderboardEntry",
    "LeaderboardMetadata",
    "Method",
    "Mode",
    "Pool",
    "REFERENCE_LABEL",
    "RecallStats",
    "ScoreRecord",
    "SourceRecord",
    "Submission",
    "TokenUsage",
    "WebSearchTrace",
    "keep_in_pool",
    # domain
    "ALL_DOMAINS",
    "DOMAINS",
    "DomainConfig",
    "REAL_WORLD_DOMAINS",
    "RESEARCH_DOMAINS",
    "get_domain",
    # paths
    "ARTIFACTS_ROOT",
    "arena_leaderboard_path",
    "arena_matches_path",
    "arena_pool_leaderboard_path",
    "cases_path",
    "score_leaderboard_path",
    "score_pool_leaderboard_path",
    "score_records_path",
    "source_dir",
    "submission_path",
    "submissions_glob",
    "summary_path",
    # io
    "io",
    # transport
    "CONSTRUCTION_REGISTRY",
    "GENERATION_REGISTRY",
    "build_chat_agent",
    "build_responses_agent",
    "configure_runtime",
]
