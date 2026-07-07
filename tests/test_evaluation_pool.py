"""Pool-filtering in arena/score aggregation: reference is always kept as anchor."""

from __future__ import annotations

import pytest

from basics import (
    ArenaMatch,
    JudgeVerdict,
    REFERENCE_LABEL,
    ScoreRecord,
    get_domain,
)
from evaluation import (
    build_arena_pool_leaderboard,
    build_score_pool_leaderboard,
    filter_matches_by_pool,
    filter_records_by_pool,
)


def _match(a: str, b: str, *, score_fwd: float = 0.75, score_rev: float = 0.25) -> ArenaMatch:
    return ArenaMatch(
        case_id="x",
        model_a=a,
        model_b=b,
        judge="j",
        forward=JudgeVerdict(winner="a" if score_fwd > 0.5 else "b", score=score_fwd),
        reverse=JudgeVerdict(winner="b" if score_rev < 0.5 else "a", score=score_rev),
    )


def _matches_fixture() -> list[ArenaMatch]:
    """All pairs among {ref, baseline:m1, baseline:m2, agent:m1, agent:m2}."""
    labels = [
        REFERENCE_LABEL,
        "baseline:m1",
        "baseline:m2",
        "agent:m1",
        "agent:m2",
    ]
    out: list[ArenaMatch] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            out.append(_match(labels[i], labels[j]))
    return out


def test_filter_matches_baseline_pool_keeps_ref_plus_baselines():
    matches = _matches_fixture()
    sub = filter_matches_by_pool(matches, "baseline")
    label_pairs = {(m.model_a, m.model_b) for m in sub}
    assert label_pairs == {
        (REFERENCE_LABEL, "baseline:m1"),
        (REFERENCE_LABEL, "baseline:m2"),
        ("baseline:m1", "baseline:m2"),
    }


def test_filter_matches_agent_pool_keeps_ref_plus_agents():
    matches = _matches_fixture()
    sub = filter_matches_by_pool(matches, "agent")
    label_pairs = {(m.model_a, m.model_b) for m in sub}
    assert label_pairs == {
        (REFERENCE_LABEL, "agent:m1"),
        (REFERENCE_LABEL, "agent:m2"),
        ("agent:m1", "agent:m2"),
    }


def test_filter_matches_full_pool_is_identity():
    matches = _matches_fixture()
    assert len(filter_matches_by_pool(matches, "full")) == len(matches)


def test_filter_matches_full_pool_returns_same_object():
    """Performance contract: pool=full short-circuits and returns the input list."""
    matches = _matches_fixture()
    assert filter_matches_by_pool(matches, "full") is matches


def test_filter_records_full_pool_returns_same_object():
    records = [
        ScoreRecord(case_id="x", model="baseline:m", judge="j",
                    pair_scores=[{"grounding": 4.0, "insight": 4.0, "justification": 4.0}]),
    ]
    assert filter_records_by_pool(records, "full") is records


def test_compute_btd_logs_warning_on_non_convergence(caplog):
    """When max_iter is too small to converge, BTD should still return ratings + warn."""
    import logging
    from evaluation.arena import compute_btd

    matches = _matches_fixture()
    with caplog.at_level(logging.WARNING, logger="hypo.evaluation.arena"):
        ratings = compute_btd(matches, max_iter=1, tol=1e-12)
    assert ratings                                              # still returns something
    assert any("did not converge" in rec.message for rec in caplog.records)


def test_compute_btd_no_warning_on_convergence(caplog):
    """Well-conditioned input with default max_iter must not log a non-convergence warning."""
    import logging
    from evaluation.arena import compute_btd

    matches = _matches_fixture()
    with caplog.at_level(logging.WARNING, logger="hypo.evaluation.arena"):
        compute_btd(matches)                                    # default max_iter=1000
    assert not any("did not converge" in rec.message for rec in caplog.records)


def test_build_arena_pool_leaderboard_sets_metadata_pool_and_includes_reference():
    domain = get_domain("biomedical_science")
    lb = build_arena_pool_leaderboard(
        config="gpt-5.4",
        domain=domain,
        judge_profile="mimo-v2-pro",
        matches=_matches_fixture(),
        pool="baseline",
    )
    assert lb.metadata.pool == "baseline"
    models = {e.model for e in lb.rankings}
    assert REFERENCE_LABEL in models
    assert "baseline:m1" in models and "baseline:m2" in models
    assert "agent:m1" not in models and "agent:m2" not in models


def test_build_arena_pool_leaderboard_n_observations_matches_filter():
    """n_observations in metadata must reflect the pool-filtered match count, not the full set."""
    domain = get_domain("biomedical_science")
    all_matches = _matches_fixture()
    lb = build_arena_pool_leaderboard(
        config="gpt-5.4", domain=domain, judge_profile="mimo-v2-pro",
        matches=all_matches, pool="agent",
    )
    assert lb.metadata.n_observations == 3  # ref-m1, ref-m2, m1-m2


def _score_record(model: str, q: float = 4.0) -> ScoreRecord:
    return ScoreRecord(
        case_id="x",
        model=model,
        judge="j",
        pair_scores=[{"grounding": q, "insight": q, "justification": q}],
    )


def test_filter_score_records_by_pool_keeps_reference():
    records = [
        _score_record(REFERENCE_LABEL),
        _score_record("baseline:m1"),
        _score_record("agent:m1"),
    ]
    kept = {r.model for r in filter_records_by_pool(records, "baseline")}
    assert kept == {REFERENCE_LABEL, "baseline:m1"}


def test_build_score_pool_leaderboard_sets_metadata_pool():
    domain = get_domain("biomedical_science")
    records = [
        _score_record(REFERENCE_LABEL, q=4.5),
        _score_record("baseline:m1", q=3.0),
        _score_record("agent:m1", q=5.0),
    ]
    lb = build_score_pool_leaderboard(
        config="gpt-5.4", domain=domain, judge_profile="mimo-v2-pro",
        records=records, pool="baseline",
    )
    assert lb.metadata.pool == "baseline"
    models = {e.model for e in lb.rankings}
    assert models == {REFERENCE_LABEL, "baseline:m1"}
