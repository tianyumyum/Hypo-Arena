"""End-to-end smoke check: imports, prompt assembly, schema round-trips, registries."""

from __future__ import annotations

import sys
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import CONSTRUCTION_REGISTRY, GENERATION_REGISTRY
from basics import (
    ALL_DOMAINS,
    REAL_WORLD_DOMAINS,
    RESEARCH_DOMAINS,
    BenchmarkCase,
    DOMAINS,
    HypothesisItem,
    RecallStats,
    Submission,
    get_domain,
    io,
)
from construction.prompts import (
    context_audit_instructions,
    context_audit_prompt,
    context_forge_initial_prompt,
    context_forge_instructions,
    context_revision_prompt,
    hypothesis_audit_instructions,
    hypothesis_audit_prompt,
    hypothesis_forge_initial_prompt,
    hypothesis_forge_instructions,
    hypothesis_revision_prompt,
)
from evaluation import (
    PAIR_KEYS,
    SET_KEYS,
    arena,
    score,
)
from evaluation.prompts import (
    arena_judge_instructions,
    arena_judge_prompt,
    score_judge_instructions,
    score_judge_prompt,
)
from generation import SKILLS, SKILL_NAMES
from generation.prompts import (
    agent_final_prompt,
    agent_intermediate_instructions,
    agent_intermediate_prompt,
    agent_writer_instructions,
    base_writer_instructions,
    base_writer_prompt,
    selector_prompt,
    skill_selector_instructions,
)


def _check_domain_inventory() -> None:
    expected = {
        "biomedical_science",
        "machine_learning",
        "social_science",
        "financial_analysis",
        "it_operations",
        "safety_investigation",
    }
    assert set(ALL_DOMAINS) == expected, ALL_DOMAINS
    assert set(RESEARCH_DOMAINS) == {
        "biomedical_science", "machine_learning", "social_science",
    }
    assert set(REAL_WORLD_DOMAINS) == {
        "financial_analysis", "it_operations", "safety_investigation",
    }
    print(f"[OK] domain inventory: {len(ALL_DOMAINS)} domains")


def _check_registries() -> None:
    assert "gpt-5.4-high" in CONSTRUCTION_REGISTRY
    assert "seed-2.0-pro" in GENERATION_REGISTRY
    assert "gpt-5.4-high" in GENERATION_REGISTRY
    print(
        f"[OK] registries: construction={len(CONSTRUCTION_REGISTRY)} "
        f"generation={len(GENERATION_REGISTRY)}"
    )


def _check_skills() -> None:
    assert len(SKILL_NAMES) >= 10, SKILL_NAMES
    for name, skill in SKILLS.items():
        assert skill.body, f"empty body for skill {name!r}"
        assert skill.description, f"empty description for skill {name!r}"
        # Output Mapping must defer to domain instructions, not the legacy hypoarena typology.
        assert "Primary finding" not in skill.body, f"{name}: stale 'Primary finding' in Output Mapping"
        assert "Systemic insight" not in skill.body, f"{name}: stale 'Systemic insight' in Output Mapping"
        assert "Actionable recommendation" not in skill.body, f"{name}: stale 'Actionable recommendation' in Output Mapping"
    print(f"[OK] skills: {len(SKILLS)} loaded ({', '.join(sorted(SKILLS)[:4])}, ...)")


def _check_construction_generation_schema_parity() -> None:
    """Construction Forge schemas and Generation candidate schemas must share field shape."""
    from construction.schema import (
        CategorizedHypothesis, HypothesisDraft, HypothesisSetDraft,
    )
    from generation.schema import (
        CategorizedHypothesisCandidate, HypothesisCandidate, HypothesisCandidateSet,
    )
    assert set(HypothesisDraft.model_fields) == set(HypothesisCandidate.model_fields)
    assert set(CategorizedHypothesis.model_fields) == set(CategorizedHypothesisCandidate.model_fields)
    assert set(HypothesisSetDraft.model_fields) == set(HypothesisCandidateSet.model_fields)
    cs = CategorizedHypothesis.model_json_schema()["properties"]["category"]
    gs = CategorizedHypothesisCandidate.model_json_schema()["properties"]["category"]
    assert cs.get("minLength") == 1 and gs.get("minLength") == 1, "category min_length=1 enforcement missing"
    print("[OK] construction/generation schema parity (category min_length=1 enforced)")


def _check_prompts_for_one_domain(domain_name: str) -> None:
    domain = get_domain(domain_name)
    record_path = io.source_metadata_path(domain_name)
    sources = io.load_sources(domain_name)
    assert sources, f"no source records for {domain_name} (looked in {record_path})"
    record = sources[0]

    cf_inst = context_forge_instructions(domain)
    ca_inst = context_audit_instructions(domain)
    hf_inst = hypothesis_forge_instructions(domain)
    ha_inst = hypothesis_audit_instructions(domain)
    assert "HypoArena" in cf_inst
    assert "HypoArena" in ca_inst
    assert "HypoArena" in hf_inst
    assert "HypoArena" in ha_inst

    _ = context_forge_initial_prompt(record)
    _ = hypothesis_forge_initial_prompt("dummy context", record)
    from basics import AuditIssue, AuditResult
    failing_audit = AuditResult(
        passed=False,
        summary="needs work",
        issues=[
            AuditIssue(
                problem="too thin", revision_instruction="add depth",
                why_it_matters="cannot support hypothesis",
            )
        ],
    )
    _ = context_revision_prompt(failing_audit)
    _ = hypothesis_revision_prompt(failing_audit)
    _ = context_audit_prompt("dummy context", record)
    _ = hypothesis_audit_prompt("dummy context", {"hypothesis": "h", "evidence": "e"}, record)

    base_inst = base_writer_instructions(domain)
    sel_inst = skill_selector_instructions(domain)
    skill_body = next(iter(SKILLS.values())).body
    agent_inst = agent_writer_instructions(domain, skill_framework=skill_body)
    assert "Baseline Mode" in base_inst
    assert "Active Analytical Methodology" in agent_inst
    if domain.multi_hypothesis and domain.category_labels:
        for label in domain.category_labels:
            assert label in base_inst, f"{domain_name}: label {label!r} missing from base_writer menu"
            assert label in agent_inst, f"{domain_name}: label {label!r} missing from agent_writer menu"
    _ = base_writer_prompt("dummy context")
    _ = selector_prompt("dummy context", list(SKILLS.values()))

    # Multi-skill pipeline assembly: each intermediate stage gets only its own skill framework;
    # final stage gets only the last skill's framework; prior_analyses thread through user prompts.
    sample_skills = list(SKILLS.values())[:3]
    inter1 = agent_intermediate_instructions(
        domain, skill_framework=sample_skills[0].body, stage_index=1, stage_total=3,
    )
    inter2 = agent_intermediate_instructions(
        domain, skill_framework=sample_skills[1].body, stage_index=2, stage_total=3,
    )
    final3 = agent_writer_instructions(domain, skill_framework=sample_skills[2].body)
    assert sample_skills[0].body in inter1 and "stage 1 of a 3-stage" in inter1
    assert sample_skills[1].body in inter2 and "stage 2 of a 3-stage" in inter2
    assert sample_skills[1].body not in inter1, "stage 1 leaked stage 2's framework"
    assert sample_skills[0].body not in inter2, "stage 2 leaked stage 1's framework"
    assert sample_skills[2].body in final3, "final missing its framework"
    assert sample_skills[0].body not in final3 and sample_skills[1].body not in final3, \
        "final leaked an intermediate framework"
    prior = [(sample_skills[0].name, "<analysis A>"), (sample_skills[1].name, "<analysis B>")]
    final_user = agent_final_prompt("ctx body", prior)
    inter_user = agent_intermediate_prompt("ctx body", prior[:1])
    assert "<analysis A>" in final_user and "<analysis B>" in final_user, "prior analyses missing in final user prompt"
    assert "<analysis A>" in inter_user, "prior analysis missing in intermediate user prompt"

    # Base mode and Agent mode writers must see the same cardinality block (so output
    # shape is identical regardless of mode); for multi-hypothesis domains, both must
    # also expose the same category menu used by construction.
    if domain.multi_hypothesis:
        b_card_marker = "This is a real-world-track case"
    else:
        b_card_marker = "This case is research-track"
    base_card = base_inst[base_inst.find(b_card_marker):base_inst.find("Domain focus:")]
    agent_card = agent_inst[agent_inst.find(b_card_marker):agent_inst.find("Domain focus:")]
    assert base_card == agent_card, f"{domain_name}: base vs agent cardinality diverges"

    if domain.multi_hypothesis and domain.category_labels:
        from construction.prompts import _real_world_cardinality as construction_card_fn
        from generation.prompts import _real_world_cardinality as generation_card_fn
        c_block = construction_card_fn(domain)
        g_block = generation_card_fn(domain)
        c_menu = c_block[c_block.find("Category menu for this case:"):]
        g_menu = g_block[g_block.find("Category menu for this case:"):]
        assert c_menu == g_menu, f"{domain_name}: construction vs generation menu diverges"

    arena_inst = arena_judge_instructions(domain)
    score_inst = score_judge_instructions(domain)
    score_inst_recall = score_judge_instructions(domain, with_recall=True)
    assert "5-level verdict" in arena_inst
    assert "1–5" in score_inst
    assert "recall" not in score_inst.lower(), "recall block leaked into default score instructions"
    # Judge calibration must call out style-bias and (for multi) explicit category lane usage.
    assert "Stylistic similarity" in arena_inst, "arena role missing style-bias calibration"
    if domain.multi_hypothesis:
        assert "category" in hf_inst, "multi-hypothesis forge instructions must require category"
        assert "category" in ha_inst, "multi-hypothesis audit instructions must mention category"
        assert "recall" in score_inst_recall.lower()
        assert "hits/total" in score_inst_recall
        assert "[category: ...]" in arena_inst, "arena role missing category-lane usage hint"
        assert "[category: ...]" in score_inst, "score role missing category-lane usage hint"
        assert "score each Hypothesis on its own merits" in score_inst, "score role missing per-pair scoring instruction"
        assert "omit `distinctness`" in score_inst, "score role missing K=1 distinctness omission"
        assert "pair-level quality" in arena_inst and "set-level quality" in arena_inst, \
            "arena role missing pair vs set weigh guidance"
    items = [HypothesisItem(hypothesis="h", evidence="e")]
    items_cat = [HypothesisItem(category="cat_a", hypothesis="h", evidence="e")]
    arena_user = arena_judge_prompt(context="ctx", submission_a=items_cat, submission_b=items_cat)
    score_user = score_judge_prompt(context="ctx", submission=items_cat)
    assert "[category: cat_a]" in arena_user, "arena user prompt does not surface category"
    assert "[category: cat_a]" in score_user, "score user prompt does not surface category"
    _ = arena_judge_prompt(context="ctx", submission_a=items, submission_b=items)
    _ = score_judge_prompt(context="ctx", submission=items)
    _ = score_judge_prompt(context="ctx", reference=items, submission=items)
    print(f"[OK] prompts/{domain_name}")


def _check_evaluation_logic() -> None:
    from basics import ArenaMatch, JudgeVerdict, ScoreRecord

    one_pair = [{k: 4.0 for k in PAIR_KEYS}]
    three_pairs = [
        {"grounding": 5.0, "insight": 5.0, "justification": 5.0},
        {"grounding": 2.0, "insight": 2.0, "justification": 2.0},
        {"grounding": 3.0, "insight": 3.0, "justification": 3.0},
    ]
    set_scores = {k: 3.0 for k in SET_KEYS}
    record_singleton = ScoreRecord(
        case_id="x",
        judge="judge",
        model="m",
        pair_scores=one_pair,
    )
    assert record_singleton.set_summary is None
    assert record_singleton.overall_score == 4.0
    assert record_singleton.recall is None
    record_multi = ScoreRecord(
        case_id="x",
        judge="judge",
        model="m",
        pair_scores=three_pairs,
        recall=RecallStats(hits=3, total=4),
        set_scores=set_scores,
    )
    # Paper §3.1.2: q_i per pair then Q_pair = mean of q_i → (5+2+3)/3 = 10/3
    assert abs(record_multi.pair_summary - 10.0 / 3.0) < 1e-9
    assert abs(record_multi.set_summary - 3.0) < 1e-9
    assert abs(record_multi.overall_score - (10.0 / 3.0 + 3.0) / 2.0) < 1e-9
    assert record_multi.recall is not None
    assert abs(record_multi.recall.ratio - 0.75) < 1e-9

    forward = JudgeVerdict(rationale="A wins", score=arena.VERDICT_SCORE["A>B"], winner="a")
    reverse = JudgeVerdict(rationale="A still wins", score=arena.VERDICT_SCORE["B>A"], winner="b")
    match = ArenaMatch(
        case_id="x",
        forward=forward,
        judge="judge",
        model_a="alpha",
        model_b="beta",
        reverse=reverse,
    )
    assert match.consistent
    assert match.debiased_score > 0.5

    ratings = arena.compute_btd([match, ArenaMatch(
        case_id="y",
        forward=JudgeVerdict(rationale="", score=arena.VERDICT_SCORE["A>>B"], winner="a"),
        judge="judge",
        model_a="alpha",
        model_b="beta",
        reverse=JudgeVerdict(rationale="", score=arena.VERDICT_SCORE["B>>A"], winner="b"),
    )])
    assert ratings["alpha"] > ratings["beta"]
    print("[OK] evaluation arithmetic + BTD")


def _check_schema_roundtrip() -> None:
    case = BenchmarkCase.model_validate_json(
        '{"context":"c","domain":"biomedical_science","hypotheses":[{"hypothesis":"h","evidence":"e"}],'
        '"id":"biomedical_science:demo","metadata":{},'
        '"provenance":{"context_rounds":1,"hypothesis_rounds":1,"profile":"gpt-5.4-high"},'
        '"quality":{"context_audit":{"passed":true,"summary":""},"hypothesis_audit":{"passed":true,"summary":""}}}'
    )
    assert case.quality.passed
    submission = Submission(
        domain=case.domain, hypotheses=case.hypotheses, id=case.id,
        provenance={"mode": "baseline", "profile": "demo"},
    )
    encoded = submission.model_dump_json()
    Submission.model_validate_json(encoded)
    print("[OK] schema round-trip")


def main() -> int:
    _check_domain_inventory()
    _check_registries()
    _check_skills()
    _check_construction_generation_schema_parity()
    _check_evaluation_logic()
    _check_schema_roundtrip()
    for domain_name in ALL_DOMAINS:
        _check_prompts_for_one_domain(domain_name)
    print("\nAll smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
