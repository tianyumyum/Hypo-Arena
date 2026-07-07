"""Render Appendix Inter-Judge Triangulation table.

Uses both judges (seed-2.0-pro headline + mimo-v2-pro independent) to demonstrate
that arena rankings are not idiosyncratic to a single judge model. Reports:

  * Per-domain Spearman ρ + Kendall τ between the two judges' baseline-pool BTD
    rankings (n = #baseline models per domain).
  * Per-domain pairwise verdict agreement rate (fraction of (case, model_a,
    model_b) matches where both judges pick the same `forward.winner`).
  * Per-domain Top-3 set agreement (overlap of top-3 model identities).
  * Per-domain Mean Absolute Rank Difference (average rank shift per model
    when switching judges).
  * Pooled cross-domain rank correlation across all (model × domain) cells,
    plus a bootstrap 95% CI on the pooled Spearman ρ (reported in the caption).

Output: tables/arena-arena.tex

Usage:
  uv run python -m scripts.paper_arena_vs_arena
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import REFERENCE_LABEL
from basics.io import load_arena_matches
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    DOMAIN_FULL_LABEL,
    ORDERED_DOMAINS,
    PAPER_MAIN_TABLE_MODELS,
    PAPER_TABLES_DIR,
    load_arena_leaderboard,
    patch_table_body,
    write_text_artifact,
)
from scripts._paper_stats import bootstrap_percentile_ci, kendall_tau, spearman_rho


SEED = "seed-2.0-pro"
MIMO = "mimo-v2-pro"
TOP_K = 5


def _eligible_baseline_labels(whitelist: tuple[str, ...]) -> set[str]:
    return {f"baseline:{p}" for p in whitelist} | {REFERENCE_LABEL}


def _ratings_dict(
    domain: str, judge: str, config: str, eligible: set[str],
) -> dict[str, float]:
    return {r.raw_label: r.rating for r in
            load_arena_leaderboard(domain, judge=judge, config=config, pool="baseline")
            if r.raw_label in eligible}


def _ranks_dict(ratings: dict[str, float]) -> dict[str, int]:
    """Rank 1 = best (highest rating); ties get averaged ranks."""
    labels_sorted = sorted(ratings, key=lambda k: -ratings[k])
    ranks: dict[str, int] = {}
    for i, label in enumerate(labels_sorted):
        ranks[label] = i + 1
    return ranks


def per_domain_btd_correlation(
    domain: str, judge_a: str, judge_b: str, config: str, eligible: set[str],
) -> tuple[float, float, int]:
    """Spearman ρ + Kendall τ between two judges' baseline-pool BTD ratings."""
    rows_a = _ratings_dict(domain, judge_a, config, eligible)
    rows_b = _ratings_dict(domain, judge_b, config, eligible)
    common = sorted(set(rows_a) & set(rows_b))
    if len(common) < 3:
        return float("nan"), float("nan"), len(common)
    xs = [rows_a[k] for k in common]
    ys = [rows_b[k] for k in common]
    return spearman_rho(xs, ys), kendall_tau(xs, ys), len(common)


def per_domain_top_k_overlap(
    domain: str, judge_a: str, judge_b: str, config: str, eligible: set[str],
    k: int,
) -> tuple[int, int]:
    """Return (|top_k_a ∩ top_k_b|, k). Intersection size over k."""
    rows_a = _ratings_dict(domain, judge_a, config, eligible)
    rows_b = _ratings_dict(domain, judge_b, config, eligible)
    top_a = set(sorted(rows_a, key=lambda kk: -rows_a[kk])[:k])
    top_b = set(sorted(rows_b, key=lambda kk: -rows_b[kk])[:k])
    return len(top_a & top_b), k


def per_domain_mard(
    domain: str, judge_a: str, judge_b: str, config: str, eligible: set[str],
) -> tuple[float, int]:
    """Mean Absolute Rank Difference over the common baseline models."""
    rows_a = _ratings_dict(domain, judge_a, config, eligible)
    rows_b = _ratings_dict(domain, judge_b, config, eligible)
    common = sorted(set(rows_a) & set(rows_b))
    if len(common) < 2:
        return float("nan"), len(common)
    ranks_a = _ranks_dict({k: rows_a[k] for k in common})
    ranks_b = _ranks_dict({k: rows_b[k] for k in common})
    diffs = [abs(ranks_a[m] - ranks_b[m]) for m in common]
    return sum(diffs) / len(diffs), len(diffs)


def pairwise_agreement_rate(
    domain: str, judge_a: str, judge_b: str, config: str, eligible: set[str],
) -> tuple[float, int]:
    """Fraction of shared matches where both judges' forward.winner agree.

    A match is keyed by (case_id, model_a, model_b). Only matches where BOTH
    judges have a verdict AND both model labels are in `eligible` count.
    """
    matches_a = {(m.case_id, m.model_a, m.model_b): m.forward.winner
                 for m in load_arena_matches(domain, config, judge_a)
                 if m.model_a in eligible and m.model_b in eligible}
    matches_b = {(m.case_id, m.model_a, m.model_b): m.forward.winner
                 for m in load_arena_matches(domain, config, judge_b)
                 if m.model_a in eligible and m.model_b in eligible}
    shared = matches_a.keys() & matches_b.keys()
    if not shared:
        return float("nan"), 0
    agree = sum(1 for k in shared if matches_a[k] == matches_b[k])
    return agree / len(shared), len(shared)


def pooled_btd_paired(
    judge_a: str, judge_b: str, config: str, eligible: set[str],
) -> tuple[list[float], list[float]]:
    """Return paired (xs, ys) of BTD ratings stacked over all (model × domain) cells."""
    xs: list[float] = []
    ys: list[float] = []
    for domain in ORDERED_DOMAINS:
        rows_a = _ratings_dict(domain, judge_a, config, eligible)
        rows_b = _ratings_dict(domain, judge_b, config, eligible)
        for label in sorted(set(rows_a) & set(rows_b)):
            xs.append(rows_a[label])
            ys.append(rows_b[label])
    return xs, ys


def pooled_btd_correlation(
    judge_a: str, judge_b: str, config: str, eligible: set[str],
) -> tuple[float, float, int]:
    """Spearman ρ + Kendall τ over all (model × domain) cells stacked."""
    xs, ys = pooled_btd_paired(judge_a, judge_b, config, eligible)
    if len(xs) < 3:
        return float("nan"), float("nan"), len(xs)
    return spearman_rho(xs, ys), kendall_tau(xs, ys), len(xs)


def _build_body(per_domain: list[dict], pooled: dict) -> str:
    body_lines: list[str] = []
    for d in per_domain:
        body_lines.append(
            f"{DOMAIN_FULL_LABEL[d['domain']]} & "
            f"{d['rho']:.3f} & {d['tau']:.3f} & "
            f"{d['mard']:.2f} & "
            f"{d['top_k_overlap']}/{d['top_k']} & "
            f"{d['agreement']*100:.1f}\\% \\\\"
        )
    body_lines.append("\\midrule")
    body_lines.append(
        f"\\textbf{{Overall}} & "
        f"\\textbf{{{pooled['rho']:.3f}}} & \\textbf{{{pooled['tau']:.3f}}} & "
        f"\\textbf{{{pooled['mard_mean']:.2f}}} & "
        f"\\textbf{{{pooled['top_k_overlap_mean']:.2f}/{pooled['top_k']}}} & "
        f"\\textbf{{{pooled['agreement']*100:.1f}\\%}} \\\\"
    )
    return "\n".join(body_lines)


def render_table(
    per_domain: list[dict],
    pooled: dict,
    judge_a: str,
    judge_b: str,
) -> str:
    body = _build_body(per_domain, pooled)
    n_models_per_domain = per_domain[0]['n_models'] if per_domain else 0
    k = pooled['top_k']
    ci_lo, ci_hi = pooled['rho_ci']
    ci_text = (
        f" Pooled $\\rho$ has bootstrap 95\\% CI $[{ci_lo:.3f}, {ci_hi:.3f}]$."
        if ci_lo == ci_lo and ci_hi == ci_hi else ""
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Inter-judge agreement between \\texttt{" + judge_a + "} and "
        "\\texttt{" + judge_b + "} on the "
        f"{n_models_per_domain} baseline-mode submissions shared by both judges "
        "in each domain. Spearman $\\rho$ and Kendall $\\tau$ are rank "
        "correlations between the two judges' BTD ratings; Rank Shift is the "
        "mean $|\\Delta\\text{rank}|$ per model; "
        f"Top-{k} is the intersection size of each judge's top-{k} models; "
        "Pairwise is the fraction of (case, model$_a$, model$_b$) matches with "
        "matching forward verdicts (chance baseline $\\approx 33\\%$). Overall "
        f"pools all six domains.{ci_text}}}\n"
        "\\label{tab:arena-arena}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{6pt}\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "\\textbf{Domain} & \\textbf{Spearman $\\rho$ ($\\uparrow$)} & "
        "\\textbf{Kendall $\\tau$ ($\\uparrow$)} & "
        "\\textbf{Rank Shift ($\\downarrow$)} & "
        "\\textbf{Top-" + str(k) + " ($\\uparrow$)} & "
        "\\textbf{Pairwise ($\\uparrow$)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--judge-a", default=SEED, help="Headline judge")
    parser.add_argument("--judge-b", default=MIMO, help="Independent second judge")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help=f"k for top-k set agreement (default: {TOP_K})")
    parser.add_argument("--n-boot", type=int, default=2000,
                        help="bootstrap iterations for ρ CI (default: 2000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for bootstrap reproducibility (default: 42)")
    parser.add_argument("--out", default=str(PAPER_TABLES_DIR / "arena-arena.tex"))
    parser.add_argument(
        "--update-rows-only", action=argparse.BooleanOptionalAction, default=True,
        help=("Patch only the body rows of an existing target file, preserving "
              "caption / label / column headers (default). Falls back to a full "
              "write when the target file does not exist."),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    eligible = _eligible_baseline_labels(PAPER_MAIN_TABLE_MODELS)

    per_domain: list[dict] = []
    pooled_matches_total = 0
    pooled_agree = 0
    top_k_overlaps: list[int] = []
    mards: list[float] = []
    for domain in ORDERED_DOMAINS:
        rho, tau, n_models = per_domain_btd_correlation(
            domain, args.judge_a, args.judge_b, args.config, eligible,
        )
        agreement, n_matches = pairwise_agreement_rate(
            domain, args.judge_a, args.judge_b, args.config, eligible,
        )
        top_k_overlap, _ = per_domain_top_k_overlap(
            domain, args.judge_a, args.judge_b, args.config, eligible, args.top_k,
        )
        mard, _ = per_domain_mard(
            domain, args.judge_a, args.judge_b, args.config, eligible,
        )
        per_domain.append({
            "domain": domain,
            "n_models": n_models,
            "rho": rho,
            "tau": tau,
            "agreement": agreement,
            "n_matches": n_matches,
            "top_k_overlap": top_k_overlap,
            "top_k": args.top_k,
            "mard": mard,
        })
        pooled_matches_total += n_matches
        pooled_agree += agreement * n_matches
        top_k_overlaps.append(top_k_overlap)
        if mard == mard:
            mards.append(mard)

    pooled_rho, pooled_tau, n_cells = pooled_btd_correlation(
        args.judge_a, args.judge_b, args.config, eligible,
    )
    pooled_xs, pooled_ys = pooled_btd_paired(
        args.judge_a, args.judge_b, args.config, eligible,
    )
    rho_ci = bootstrap_percentile_ci(
        pooled_xs, pooled_ys, spearman_rho,
        n_boot=args.n_boot, seed=args.seed,
    )
    pooled = {
        "rho": pooled_rho,
        "tau": pooled_tau,
        "rho_ci": rho_ci,
        "n_cells": n_cells,
        "agreement": pooled_agree / pooled_matches_total if pooled_matches_total else float("nan"),
        "n_matches": pooled_matches_total,
        "top_k": args.top_k,
        "top_k_overlap_mean": sum(top_k_overlaps) / len(top_k_overlaps) if top_k_overlaps else float("nan"),
        "mard_mean": sum(mards) / len(mards) if mards else float("nan"),
    }

    target = Path(args.out)
    if args.update_rows_only and target.exists():
        patch_table_body(target, _build_body(per_domain, pooled))
    else:
        text = render_table(per_domain, pooled, args.judge_a, args.judge_b)
        write_text_artifact(target, text)

    print()
    print(f"Per-domain inter-judge agreement (judges: {args.judge_a} vs {args.judge_b}, top_k={args.top_k}):")
    for d in per_domain:
        print(f"  {DOMAIN_FULL_LABEL[d['domain']]:<22s}  "
              f"n_models={d['n_models']:>2d}  "
              f"ρ={d['rho']:.3f}  τ={d['tau']:.3f}  "
              f"shift={d['mard']:.2f}  "
              f"top-{d['top_k']}={d['top_k_overlap']}/{d['top_k']}  "
              f"agree={d['agreement']*100:5.1f}%  "
              f"(n_matches={d['n_matches']})")
    print(f"  {'Pooled':<22s}  n_cells={pooled['n_cells']:>2d}  "
          f"ρ={pooled['rho']:.3f} CI[{rho_ci[0]:.3f},{rho_ci[1]:.3f}]  "
          f"τ={pooled['tau']:.3f}  "
          f"shift={pooled['mard_mean']:.2f}  "
          f"top-{pooled['top_k']}={pooled['top_k_overlap_mean']:.2f}/{pooled['top_k']}  "
          f"agree={pooled['agreement']*100:5.1f}%  "
          f"(n_matches={pooled['n_matches']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
