"""Roundtrip + path + domain sanity tests for basics/."""

from __future__ import annotations

import pytest

from basics import (
    ALL_DOMAINS,
    ARTIFACTS_ROOT,
    DOMAINS,
    REAL_WORLD_DOMAINS,
    REFERENCE_LABEL,
    RESEARCH_DOMAINS,
    ArenaMatch,
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
    RecallStats,
    ScoreRecord,
    SourceRecord,
    Submission,
    TokenUsage,
    arena_leaderboard_path,
    arena_matches_path,
    arena_pool_leaderboard_path,
    cases_path,
    get_domain,
    keep_in_pool,
    score_leaderboard_path,
    score_pool_leaderboard_path,
    score_records_path,
    source_dir,
    submission_path,
    summary_path,
)


def _sample_case() -> BenchmarkCase:
    return BenchmarkCase(
        id="biomedical_science:pmid-12345",
        domain="biomedical_science",
        metadata={"title": "Test paper", "doi": "10.0/test"},
        context="A randomized trial compared X vs Y in 100 patients...",
        hypotheses=[
            HypothesisItem(hypothesis="X works better", evidence="0.8% lower HbA1c, p=0.003"),
        ],
        quality=CaseQuality(
            context_audit=AuditResult(passed=True, summary="ok"),
            hypothesis_audit=AuditResult(passed=True, summary="ok"),
        ),
        provenance=ConstructionProvenance(
            profile="gpt-5.4-high",
            context_rounds=2,
            hypothesis_rounds=1,
            tokens=TokenUsage(requests=2, input_tokens=1000, output_tokens=300, cached_tokens=500, reasoning_tokens=100),
        ),
    )


# ---- schema roundtrip ----

def test_source_record_roundtrip():
    rec = SourceRecord(
        id="biomedical_science:10.1016_j.celrep.2025.116174",
        domain="biomedical_science",
        title="",
        file="10.1016/j.celrep.2025.116174.pdf",
        metadata={"doi": "https://doi.org/10.1016/j.celrep.2025.116174"},
    )
    restored = SourceRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec
    assert restored.url is None


def test_source_record_url_only():
    rec = SourceRecord(
        id="it_operations:doc_0SRaIuQV",
        domain="it_operations",
        title="Google Cloud Service Health",
        url="https://status.cloud.google.com/incidents/RmPhfQT9RDGwWLCXS2sC",
        metadata={"report_id": "doc_0SRaIuQV"},
    )
    restored = SourceRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec
    assert restored.file is None


def test_benchmark_case_roundtrip():
    case = _sample_case()
    restored = BenchmarkCase.model_validate_json(case.model_dump_json())
    assert restored == case
    assert restored.schema_version == 1


def test_case_quality_passed_is_computed():
    quality = CaseQuality(
        context_audit=AuditResult(passed=True, summary="ok"),
        hypothesis_audit=AuditResult(passed=False, summary="bad"),
    )
    assert quality.passed is False
    quality_pass = CaseQuality(
        context_audit=AuditResult(passed=True, summary="ok"),
        hypothesis_audit=AuditResult(passed=True, summary="ok"),
    )
    assert quality_pass.passed is True


def test_submission_roundtrip():
    sub = Submission(
        id="biomedical_science:pmid-12345",
        domain="biomedical_science",
        hypotheses=[HypothesisItem(hypothesis="alt", evidence="trial data")],
        provenance=GenerationProvenance(
            profile="claude-opus-4.6-high",
            mode="baseline",
            tokens=TokenUsage(requests=1, input_tokens=2000, output_tokens=500),
            fallback_platform="platform_a",
        ),
    )
    restored = Submission.model_validate_json(sub.model_dump_json())
    assert restored == sub


def test_arena_match_roundtrip_and_computed():
    match = ArenaMatch(
        case_id="biomedical_science:pmid-12345",
        model_a="gpt-5.4-high",
        model_b="claude-opus-4.6-high",
        judge="grok-4.1-fast",
        forward=JudgeVerdict(winner="a", score=0.7, rubric_scores={"grounding": 4.0}),
        reverse=JudgeVerdict(winner="b", score=0.3, rubric_scores={"grounding": 4.0}),
    )
    # Forward says A wins; reverse says B (= A in original positions) wins → consistent.
    assert match.consistent is True
    assert match.debiased_score == pytest.approx((0.7 + (1 - 0.3)) / 2)
    restored = ArenaMatch.model_validate_json(match.model_dump_json())
    assert restored == match
    assert restored.consistent is True


def test_arena_match_inconsistent_when_both_pick_position_a():
    match = ArenaMatch(
        case_id="x", model_a="m1", model_b="m2", judge="j",
        forward=JudgeVerdict(winner="a", score=0.6),
        reverse=JudgeVerdict(winner="a", score=0.6),   # position bias
    )
    assert match.consistent is False


def test_arena_match_consistent_when_both_tie():
    match = ArenaMatch(
        case_id="x", model_a="m1", model_b="m2", judge="j",
        forward=JudgeVerdict(winner="tie", score=0.5),
        reverse=JudgeVerdict(winner="tie", score=0.5),
    )
    assert match.consistent is True


def test_score_record_roundtrip_and_computed():
    rec = ScoreRecord(
        case_id="biomedical_science:pmid-12345",
        model="gpt-5.4-high",
        judge="grok-4.1-fast",
        pair_scores=[{"grounding": 4.5, "insight": 3.5, "justification": 4.0}],
        set_scores={},
    )
    assert rec.pair_summary == pytest.approx(4.0)
    assert rec.set_summary is None
    assert rec.overall_score == pytest.approx(4.0)
    restored = ScoreRecord.model_validate_json(rec.model_dump_json())
    assert restored == rec
    assert restored.overall_score == pytest.approx(4.0)


def test_score_record_multi_hypothesis_aggregates():
    rec = ScoreRecord(
        case_id="x", model="m", judge="j",
        pair_scores=[
            {"grounding": 5.0, "insight": 5.0, "justification": 5.0},
            {"grounding": 2.0, "insight": 2.0, "justification": 2.0},
            {"grounding": 3.0, "insight": 3.0, "justification": 3.0},
        ],
        set_scores={"breadth": 3.0, "distinctness": 3.0, "utility": 3.0},
    )
    # q_i per pair then Q_pair = mean of q_i → (5+2+3)/3
    assert rec.pair_summary == pytest.approx(10.0 / 3.0)
    assert rec.set_summary == pytest.approx(3.0)
    assert rec.overall_score == pytest.approx((10.0 / 3.0 + 3.0) / 2.0)


def test_score_record_k1_distinctness_omitted_in_multi_domain():
    """When K=1 in a multi domain, distinctness is N/A and set_summary uses (b+u)/2."""
    rec = ScoreRecord(
        case_id="x", model="m", judge="j",
        pair_scores=[{"grounding": 4.0, "insight": 4.0, "justification": 4.0}],
        set_scores={"breadth": 4.0, "utility": 2.0},  # distinctness omitted
    )
    assert rec.pair_summary == pytest.approx(4.0)
    assert rec.set_summary == pytest.approx(3.0)  # (4+2)/2
    assert rec.overall_score == pytest.approx(3.5)


def test_recall_stats_ratio_is_computed():
    stats = RecallStats(hits=3, total=5)
    assert stats.ratio == pytest.approx(0.6)
    restored = RecallStats.model_validate_json(stats.model_dump_json())
    assert restored == stats


def test_recall_stats_handles_zero_total():
    assert RecallStats(hits=0, total=0).ratio == 0.0


def test_score_record_optionally_carries_recall():
    rec = ScoreRecord(
        case_id="x", model="m", judge="j",
        pair_scores=[{"grounding": 4.0, "insight": 4.0, "justification": 4.0}],
        recall=RecallStats(hits=2, total=4),
    )
    assert rec.recall is not None
    assert rec.recall.ratio == pytest.approx(0.5)
    restored = ScoreRecord.model_validate_json(rec.model_dump_json())
    assert restored.recall == rec.recall


def test_score_record_recall_defaults_to_none():
    rec = ScoreRecord(
        case_id="x", model="m", judge="j",
        pair_scores=[{"grounding": 4.0, "insight": 4.0, "justification": 4.0}],
    )
    assert rec.recall is None


def test_token_usage_addition():
    a = TokenUsage(cached_tokens=1, input_tokens=10, output_tokens=20, reasoning_tokens=3, requests=1)
    b = TokenUsage(cached_tokens=2, input_tokens=30, output_tokens=40, reasoning_tokens=5, requests=2)
    c = a + b
    assert c == TokenUsage(cached_tokens=3, input_tokens=40, output_tokens=60, reasoning_tokens=8, requests=3)
    assert c.total_tokens == 100


def test_token_usage_from_sdk_usage_duck_typed():
    class _SDK:
        input_tokens = 100
        output_tokens = 50
        requests = 1
        class _IDetails: cached_tokens = 25
        class _ODetails: reasoning_tokens = 7
        input_tokens_details = _IDetails()
        output_tokens_details = _ODetails()
    tu = TokenUsage.from_sdk_usage(_SDK())
    assert tu == TokenUsage(cached_tokens=25, input_tokens=100, output_tokens=50, reasoning_tokens=7, requests=1)


def test_token_usage_from_sdk_usage_handles_missing_details():
    class _Bare:
        input_tokens = 5
        output_tokens = 8
        requests = 1
    tu = TokenUsage.from_sdk_usage(_Bare())
    assert tu == TokenUsage(cached_tokens=0, input_tokens=5, output_tokens=8, reasoning_tokens=0, requests=1)


def test_hypothesis_item_carries_optional_category():
    bare = HypothesisItem(hypothesis="h", evidence="e")
    assert bare.category is None
    tagged = HypothesisItem(category="organizational_factor", hypothesis="h", evidence="e")
    assert tagged.category == "organizational_factor"
    restored = HypothesisItem.model_validate_json(tagged.model_dump_json())
    assert restored == tagged


def test_leaderboard_roundtrip():
    lb = Leaderboard(
        metadata=LeaderboardMetadata(
            method="arena",
            domain="biomedical_science",
            config="gpt-5.4",
            judge="grok-4.1-fast",
            n_models=3,
            n_observations=42,
            position_consistency_rate=0.85,
            btd_iterations=120,
        ),
        rankings=[
            LeaderboardEntry(rank=1, model="claude-opus-4.6-high", rating=1612.3, n_observations=14),
            LeaderboardEntry(rank=2, model="gpt-5.4-high", rating=1500.0, n_observations=14),
        ],
    )
    restored = Leaderboard.model_validate_json(lb.model_dump_json())
    assert restored == lb


def test_unknown_extra_field_ignored():
    case = _sample_case()
    payload = case.model_dump()
    payload["mystery_field"] = "future-schema-extension"
    restored = BenchmarkCase.model_validate(payload)
    assert restored == case


# ---- paths ----

def test_paths_are_relative_to_benchmark_root():
    assert source_dir("biomedical_science") == ARTIFACTS_ROOT / "biomedical_science" / "source"
    assert cases_path("financial_analysis", "gpt-5.4") == (
        ARTIFACTS_ROOT / "financial_analysis" / "cases" / "gpt-5.4.jsonl"
    )


def test_submission_path_layout():
    p = submission_path("biomedical_science", "baseline", "gpt-5.4", "claude-opus-4.6-high")
    assert p == (
        ARTIFACTS_ROOT / "biomedical_science" / "submissions" / "baseline"
        / "gpt-5.4+claude-opus-4.6-high.jsonl"
    )


def test_submissions_glob_returns_matching_files(tmp_path, monkeypatch):
    import basics.paths as paths
    monkeypatch.setattr(paths, "ARTIFACTS_ROOT", tmp_path)

    base = tmp_path / "biomedical_science" / "submissions" / "baseline"
    base.mkdir(parents=True)
    (base / "gpt-5.4+claude-opus-4.6-high.jsonl").touch()
    (base / "gpt-5.4+gpt-5.4-high.jsonl").touch()
    (base / "gpt-5.4+other-noise.txt").touch()
    (base / "different-config+x.jsonl").touch()

    found = paths.submissions_glob("biomedical_science", "baseline", "gpt-5.4")
    names = sorted(p.name for p in found)
    assert names == ["gpt-5.4+claude-opus-4.6-high.jsonl", "gpt-5.4+gpt-5.4-high.jsonl"]


def test_evaluation_path_layout():
    assert arena_matches_path("biomedical_science", "gpt-5.4", "grok-4.1-fast") == (
        ARTIFACTS_ROOT / "biomedical_science" / "results" / "gpt-5.4.grok-4.1-fast.arena.matches.jsonl"
    )
    assert arena_leaderboard_path("biomedical_science", "gpt-5.4", "grok-4.1-fast", "md") == (
        ARTIFACTS_ROOT / "biomedical_science" / "results" / "gpt-5.4.grok-4.1-fast.arena.leaderboard.md"
    )
    assert score_records_path("financial_analysis", "gpt-5.4", "seed-2.0-pro") == (
        ARTIFACTS_ROOT / "financial_analysis" / "results" / "gpt-5.4.seed-2.0-pro.score.jsonl"
    )
    assert score_leaderboard_path("financial_analysis", "gpt-5.4", "seed-2.0-pro") == (
        ARTIFACTS_ROOT / "financial_analysis" / "results" / "gpt-5.4.seed-2.0-pro.score.leaderboard.json"
    )


def test_summary_path_layout():
    assert summary_path("arena", "gpt-5.4", "grok-4.1-fast") == (
        ARTIFACTS_ROOT / "_summary" / "gpt-5.4.grok-4.1-fast.arena.summary.json"
    )


def test_pool_leaderboard_paths():
    assert arena_pool_leaderboard_path("biomedical_science", "gpt-5.4", "mimo-v2-pro", "baseline") == (
        ARTIFACTS_ROOT / "biomedical_science" / "results"
        / "gpt-5.4.mimo-v2-pro.arena.baseline.leaderboard.json"
    )
    assert arena_pool_leaderboard_path("it_operations", "gpt-5.4", "mimo-v2-pro", "agent", "md") == (
        ARTIFACTS_ROOT / "it_operations" / "results"
        / "gpt-5.4.mimo-v2-pro.arena.agent.leaderboard.md"
    )
    assert score_pool_leaderboard_path("safety_investigation", "gpt-5.4", "seed-2.0-pro", "full") == (
        ARTIFACTS_ROOT / "safety_investigation" / "results"
        / "gpt-5.4.seed-2.0-pro.score.full.leaderboard.json"
    )


def test_keep_in_pool_reference_always_kept():
    for pool in ("baseline", "agent", "full"):
        assert keep_in_pool(REFERENCE_LABEL, pool) is True


def test_keep_in_pool_baseline_excludes_agent_labels():
    assert keep_in_pool("baseline:gpt-5.4-high", "baseline") is True
    assert keep_in_pool("agent:gpt-5.4-high", "baseline") is False


def test_keep_in_pool_agent_excludes_baseline_labels():
    assert keep_in_pool("agent:gpt-5.4-high", "agent") is True
    assert keep_in_pool("baseline:gpt-5.4-high", "agent") is False


def test_keep_in_pool_full_keeps_everything():
    for label in ("baseline:m", "agent:m", REFERENCE_LABEL):
        assert keep_in_pool(label, "full") is True


def test_leaderboard_metadata_round_trip_with_pool():
    md = LeaderboardMetadata(
        config="gpt-5.4", domain="it_operations", judge="mimo-v2-pro",
        method="arena", n_models=5, n_observations=120, pool="baseline",
        position_consistency_rate=0.81,
    )
    restored = LeaderboardMetadata.model_validate_json(md.model_dump_json())
    assert restored == md
    assert restored.pool == "baseline"


def test_leaderboard_metadata_pool_optional():
    md = LeaderboardMetadata(
        config="gpt-5.4", domain="biomedical_science", judge="mimo-v2-pro",
        method="score", n_models=3, n_observations=42,
    )
    assert md.pool is None
    restored = LeaderboardMetadata.model_validate_json(md.model_dump_json())
    assert restored.pool is None


# ---- domain config ----

def test_six_domains_present():
    assert len(ALL_DOMAINS) == 6
    assert set(RESEARCH_DOMAINS) | set(REAL_WORLD_DOMAINS) == set(ALL_DOMAINS)
    assert len(RESEARCH_DOMAINS) == 3
    assert len(REAL_WORLD_DOMAINS) == 3


def test_research_domains_are_single_hypothesis():
    for name in RESEARCH_DOMAINS:
        assert DOMAINS[name].multi_hypothesis is False


def test_real_world_domains_are_multi_hypothesis():
    for name in REAL_WORLD_DOMAINS:
        assert DOMAINS[name].multi_hypothesis is True


def test_get_domain_raises_on_unknown():
    with pytest.raises(KeyError, match="Unknown domain"):
        get_domain("nonexistent_domain")
