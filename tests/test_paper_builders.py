"""Smoke tests for the seven §4.x paper builders.

Sets up a synthetic ARTIFACTS_ROOT with arena.matches.jsonl + score.jsonl across
all 6 domains, derives the 3 pool leaderboards via the same code paths the
orchestrator uses, then invokes each builder's main() with --out pointing to a
temp file. Asserts: exit code 0 + output file exists + non-empty.

These tests do NOT verify exact numeric output — that's covered by integration
with real artifacts. They protect against import errors, signature drift, and
crashes on edge cases (empty pools, single-mode models, etc.).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
import pytest

# Force headless matplotlib before any builder imports it.
matplotlib.use("Agg")

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import (
    ALL_DOMAINS,
    ArenaMatch,
    JudgeVerdict,
    REFERENCE_LABEL,
    RecallStats,
    ScoreRecord,
    arena_pool_leaderboard_path,
    get_domain,
    score_pool_leaderboard_path,
)
from basics.io import (
    append_arena_match,
    append_score_record,
    load_arena_matches,
    load_score_records,
)
from evaluation import build_arena_pool_leaderboard, build_score_pool_leaderboard
from evaluation.markdown import render_leaderboard_md
from orchestrator.atomic import write_json_atomic, write_text_atomic


_CONFIG = "gpt-5.4"
_JUDGE = "mimo-v2-pro"

# Pick profiles from the paper whitelist so build_main_results doesn't drop them.
_PROFILES = (
    "claude-sonnet-4.6-high",
    "gpt-5.4-high",
    "kimi-k2.6-thinking",
)
_LABELS = (REFERENCE_LABEL,) + tuple(f"baseline:{p}" for p in _PROFILES) \
    + tuple(f"agent:{p}" for p in _PROFILES)


def _seed_arena(domain: str, di: int) -> None:
    """Generate matches across 3 cases × all label pairs; ratings vary by index."""
    for ci in range(3):
        case_id = f"{domain}:case_{ci}"
        for i in range(len(_LABELS)):
            for j in range(i + 1, len(_LABELS)):
                # Make label[0]=reference always lose half, stronger label wins.
                fwd_score = 0.65 - 0.05 * (j - i) + 0.005 * di
                fwd_score = max(0.05, min(0.95, fwd_score))
                rev_score = 1.0 - fwd_score
                match = ArenaMatch(
                    case_id=case_id,
                    model_a=_LABELS[i],
                    model_b=_LABELS[j],
                    judge=_JUDGE,
                    forward=JudgeVerdict(
                        winner="a" if fwd_score > 0.5 else "b", score=fwd_score),
                    reverse=JudgeVerdict(
                        winner="b" if rev_score < 0.5 else "a", score=rev_score),
                )
                append_arena_match(domain, _CONFIG, _JUDGE, match)


def _seed_score(domain: str, di: int) -> None:
    """Generate score records (with recall populated for real-world domains)."""
    domain_cfg = get_domain(domain)
    for ci in range(3):
        case_id = f"{domain}:case_{ci}"
        for li, label in enumerate(_LABELS):
            base_q = 4.5 - 0.1 * li + 0.005 * di
            pair_scores = [{
                "grounding": min(5.0, base_q),
                "insight": min(5.0, base_q + 0.1),
                "justification": min(5.0, base_q - 0.1),
            }]
            set_scores: dict[str, float] = {}
            recall = None
            if domain_cfg.multi_hypothesis:
                set_scores = {"breadth": 4.0, "distinctness": 4.0, "utility": 4.0}
                if label != REFERENCE_LABEL:
                    # Strong models recall more; reference is excluded by builder anyway.
                    hits = max(1, 5 - li)
                    recall = RecallStats(hits=hits, total=5)
            record = ScoreRecord(
                case_id=case_id, model=label, judge=_JUDGE,
                pair_scores=pair_scores, set_scores=set_scores, recall=recall,
            )
            append_score_record(domain, _CONFIG, _JUDGE, record)


def _build_pool_leaderboards(domain: str) -> None:
    """Mirror orchestrator._rebuild_*_leaderboard for one (domain, judge)."""
    domain_cfg = get_domain(domain)
    matches = load_arena_matches(domain, _CONFIG, _JUDGE)
    for pool in ("baseline", "agent", "full"):
        lb = build_arena_pool_leaderboard(
            config=_CONFIG, domain=domain_cfg, judge_profile=_JUDGE,
            matches=matches, pool=pool,
        )
        write_json_atomic(arena_pool_leaderboard_path(domain, _CONFIG, _JUDGE, pool), lb)
        write_text_atomic(
            arena_pool_leaderboard_path(domain, _CONFIG, _JUDGE, pool, suffix="md"),
            render_leaderboard_md(lb),
        )

    records = load_score_records(domain, _CONFIG, _JUDGE)
    for pool in ("baseline", "agent", "full"):
        lb = build_score_pool_leaderboard(
            config=_CONFIG, domain=domain_cfg, judge_profile=_JUDGE,
            records=records, pool=pool,
        )
        write_json_atomic(score_pool_leaderboard_path(domain, _CONFIG, _JUDGE, pool), lb)
        write_text_atomic(
            score_pool_leaderboard_path(domain, _CONFIG, _JUDGE, pool, suffix="md"),
            render_leaderboard_md(lb),
        )


@pytest.fixture
def paper_artifacts(tmp_path, monkeypatch):
    """Synthetic ARTIFACTS_ROOT with seeded matches/scores + derived pool leaderboards."""
    import basics.paths as paths_module
    monkeypatch.setattr(paths_module, "ARTIFACTS_ROOT", tmp_path)

    for di, domain in enumerate(ALL_DOMAINS):
        _seed_arena(domain, di)
        _seed_score(domain, di)
        _build_pool_leaderboards(domain)

    return tmp_path


def _run_builder(module_name: str, out_path: Path, extra: list[str] | None = None) -> None:
    """Reload the builder module fresh (so any cached default paths re-resolve), then call main()."""
    if module_name in sys.modules:
        importlib.reload(sys.modules[module_name])
    module = importlib.import_module(module_name)
    argv = ["--out", str(out_path)]
    if extra:
        argv.extend(extra)
    rc = module.main(argv)
    assert rc == 0, f"{module_name} main() returned {rc}"


def test_build_main_results(paper_artifacts, tmp_path):
    """Single-judge mode with explicit --out (the legacy invocation)."""
    out = tmp_path / "arena-seed.tex"
    _run_builder(
        "scripts.paper_arena", out,
        extra=["--judges", "mimo-v2-pro"],
    )
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "\\begin{table*}" in text and "\\end{table*}" in text


def test_build_main_results_multi_judge_writes_one_file_per_judge(paper_artifacts, tmp_path):
    """Default --judges path: one arena-<short>.tex per judge into --out-dir."""
    if "scripts.paper_arena" in sys.modules:
        importlib.reload(sys.modules["scripts.paper_arena"])
    module = importlib.import_module("scripts.paper_arena")
    rc = module.main([
        "--judges", "mimo-v2-pro",                              # fixture only seeds mimo
        "--out-dir", str(tmp_path),
    ])
    assert rc == 0
    assert (tmp_path / "arena-mimo.tex").exists()


def test_build_main_results_out_requires_single_judge(paper_artifacts, tmp_path):
    """--out + multi-judge must fail — ambiguous target file."""
    if "scripts.paper_arena" in sys.modules:
        importlib.reload(sys.modules["scripts.paper_arena"])
    module = importlib.import_module("scripts.paper_arena")
    out = tmp_path / "x.tex"
    with pytest.raises(SystemExit):
        module.main([
            "--judges", "mimo-v2-pro,seed-2.0-pro",
            "--out", str(out),
        ])


def test_fig_baseline_vs_agent(paper_artifacts, tmp_path):
    out = tmp_path / "fig_baseline_vs_agent.pdf"
    # Disable qwen-default-exclude so the synthetic models (no qwen) all show.
    _run_builder("scripts.paper_baseline_vs_agent", out, extra=["--include-all"])
    assert out.exists() and out.stat().st_size > 1000           # PDFs have nonzero overhead


def test_fig_arena_vs_score(paper_artifacts, tmp_path):
    out = tmp_path / "arena-vs-score.pdf"
    _run_builder("scripts.paper_arena_vs_score", out)
    assert out.exists() and out.stat().st_size > 1000


def test_build_recall_table(paper_artifacts, tmp_path):
    out = tmp_path / "recall.tex"
    # Lower the n>=50 threshold; synthetic data has only 9 cases × 3 real-world.
    _run_builder("scripts.paper_recall", out, extra=["--min-cases", "1"])
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "Spearman" in text and "tabular" in text


def test_build_recall_table_arena_domains_realworld(paper_artifacts, tmp_path):
    """--arena-domains realworld must change the rendered caption."""
    out = tmp_path / "recall_rw.tex"
    _run_builder(
        "scripts.paper_recall", out,
        extra=["--min-cases", "1", "--arena-domains", "realworld"],
    )
    text = out.read_text(encoding="utf-8")
    assert "averaged across realworld domains" in text


def test_build_main_results_restricted_btd(paper_artifacts, tmp_path):
    """--restricted-btd path must produce a valid table."""
    out = tmp_path / "arena-seed-restricted.tex"
    _run_builder(
        "scripts.paper_arena", out,
        extra=["--restricted-btd", "--judges", "mimo-v2-pro"],
    )
    assert out.exists() and out.stat().st_size > 0
    text = out.read_text(encoding="utf-8")
    assert "\\begin{table*}" in text


def test_build_main_results_restricted_btd_requires_whitelist(paper_artifacts, tmp_path):
    """--restricted-btd + --no-whitelist must fail with a clear error."""
    out = tmp_path / "arena-bad.tex"
    if "scripts.paper_arena" in sys.modules:
        importlib.reload(sys.modules["scripts.paper_arena"])
    module = importlib.import_module("scripts.paper_arena")
    with pytest.raises(SystemExit):
        module.main([
            "--out", str(out), "--judges", "mimo-v2-pro",
            "--restricted-btd", "--no-whitelist",
        ])
