"""ICLR 2026 Accept-vs-Reject validity check for baseline-pool arena.

Tests whether per-case Reference arena strength is higher on accepted ICLR 2026
papers (Oral + Poster) than on rejected ones, on the machine_learning domain.
Reports three complementary metrics under each configured judge:

  1. Per-case Debiased: mean per-match `debiased_score` flipped to Reference view
  2. Per-case Win Rate: 0/0.5/1 per-match AND-of-both score (Reference wins both
     forward and reverse), averaged per case
  3. Per-bucket BTD: BTD rating computed on the bucket's matches only, reading
     Reference's rating; opponent-strength-adjusted, the canonical arena currency

Output: tables/alignment-ml.tex (LaTeX, paper-ready).

Usage:
  uv run python -m scripts.paper_alignment_ml
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import BenchmarkCase
from basics.io import iter_jsonl, load_arena_matches
from basics.paths import cases_path
from basics.schema import ArenaMatch, REFERENCE_LABEL
from evaluation.arena import compute_btd, filter_matches_by_pool
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    PAPER_TABLES_DIR,
    PRIMARY_JUDGE,
    patch_table_body,
    write_text_artifact,
)
from scripts._paper_stats import bootstrap_ci_1d, mann_whitney_u

DOMAIN = "machine_learning"
ACCEPT_VENUES = ("ICLR 2026 Oral", "ICLR 2026 Poster")
REJECT_VENUES = ("ICLR 2026 Reject",)
BUCKETS = ("Accepted", "Rejected")  # display order


def _ref_debiased(m: ArenaMatch) -> float:
    s = m.debiased_score
    return 1.0 - s if m.model_b == REFERENCE_LABEL else s


def _ref_winrate(m: ArenaMatch) -> float:
    """0 / 0.5 / 1 per-match score (AND-of-both)."""
    is_ref_a = m.model_a == REFERENCE_LABEL
    is_ref_b = m.model_b == REFERENCE_LABEL
    fw, rw = m.forward.winner, m.reverse.winner
    f_win = (fw == "a" and is_ref_a) or (fw == "b" and is_ref_b)
    f_lose = (fw == "a" and not is_ref_a) or (fw == "b" and not is_ref_b)
    r_win = (rw == "a" and is_ref_b) or (rw == "b" and is_ref_a)
    r_lose = (rw == "a" and not is_ref_b) or (rw == "b" and not is_ref_a)
    if f_win and r_win:
        return 1.0
    if f_lose and r_lose:
        return 0.0
    return 0.5


def _bucket_of_venue(venue: str | None) -> str | None:
    if venue in ACCEPT_VENUES:
        return "Accepted"
    if venue in REJECT_VENUES:
        return "Rejected"
    return None


def _load_bucket_map(config: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for case in iter_jsonl(cases_path(DOMAIN, config), BenchmarkCase):
        venue = case.metadata.get("venue") if case.metadata else None
        b = _bucket_of_venue(venue)
        if b is not None:
            out[case.id] = b
    return out


def _per_case_metrics(
    matches: list[ArenaMatch],
) -> tuple[dict[str, float], dict[str, float]]:
    """Aggregate baseline-pool Reference matches into per-case (debiased, winrate)."""
    debs: dict[str, list[float]] = defaultdict(list)
    wins: dict[str, list[float]] = defaultdict(list)
    for m in matches:
        if REFERENCE_LABEL not in (m.model_a, m.model_b):
            continue
        debs[m.case_id].append(_ref_debiased(m))
        wins[m.case_id].append(_ref_winrate(m))
    deb_mean = {cid: sum(v) / len(v) for cid, v in debs.items()}
    win_mean = {cid: sum(v) / len(v) for cid, v in wins.items()}
    return deb_mean, win_mean


def _bucket_split(
    per_case: dict[str, float], bucket_of: dict[str, str],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {b: [] for b in BUCKETS}
    for cid, val in per_case.items():
        b = bucket_of.get(cid)
        if b in out:
            out[b].append(val)
    return out


def _bucket_matches(
    matches: list[ArenaMatch], bucket_of: dict[str, str],
) -> dict[str, list[ArenaMatch]]:
    out: dict[str, list[ArenaMatch]] = {b: [] for b in BUCKETS}
    for m in matches:
        b = bucket_of.get(m.case_id)
        if b in out:
            out[b].append(m)
    return out


def _per_case_summary(
    values: dict[str, list[float]], *, n_boot: int, seed: int,
) -> dict[str, dict]:
    summary: dict[str, dict] = {}
    for b in BUCKETS:
        vs = values[b]
        if vs:
            med = median(vs)
            lo, hi = bootstrap_ci_1d(vs, median, n_boot=n_boot, seed=seed)
        else:
            med, lo, hi = float("nan"), float("nan"), float("nan")
        summary[b] = {"n": len(vs), "median": med, "ci_lo": lo, "ci_hi": hi}
    if values["Accepted"] and values["Rejected"]:
        U, p = mann_whitney_u(values["Accepted"], values["Rejected"], alternative="greater")
    else:
        U, p = float("nan"), float("nan")
    summary["mwu"] = {"U": U, "p": p}
    return summary


def _btd_summary(buckets: dict[str, list[ArenaMatch]]) -> dict:
    out: dict = {}
    refs: dict[str, float] = {}
    for b in BUCKETS:
        ms = buckets[b]
        if not ms:
            out[b] = {"n_matches": 0, "ref_btd": float("nan")}
            refs[b] = float("nan")
            continue
        ratings = compute_btd(ms)
        ref_r = ratings.get(REFERENCE_LABEL, float("nan"))
        out[b] = {"n_matches": len(ms), "ref_btd": ref_r}
        refs[b] = ref_r
    out["gap"] = (refs["Accepted"] - refs["Rejected"]
                  if all(r == r for r in refs.values()) else float("nan"))
    return out


def _fmt_med(g: dict) -> str:
    if g["n"] == 0 or g["median"] != g["median"]:
        return "--"
    return f"{g['median']:.3f}"


def _fmt_btd(g: dict) -> str:
    if g.get("ref_btd") != g.get("ref_btd"):
        return "--"
    return f"{g['ref_btd']:.1f}"


def _fmt_p(p: float) -> str:
    if p != p:
        return "--"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"


def _fmt_btd_gap(gap: float) -> str:
    if gap != gap:
        return "--"
    return f"$\\Delta={'+' if gap >= 0 else ''}{gap:.1f}$"


def _build_body(deb: dict, win: dict, btd: dict) -> str:
    """Just the data rows between \\midrule and \\bottomrule."""
    rows = []
    for b in BUCKETS:
        rows.append(" & ".join([
            b,
            _fmt_med_ci(deb[b]), _fmt_med_ci(win[b]), _fmt_btd(btd[b]),
        ]) + " \\\\")
    return "\n".join(rows)


def render_table(
    judge: str, deb: dict, win: dict, btd: dict,
) -> str:
    """Single-judge table: Group | Debiased | Win Rate | BTD."""
    body = _build_body(deb, win, btd)
    n_a_cov = deb["Accepted"]["n"]
    n_r_cov = deb["Rejected"]["n"]
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Reference arena strength under \\texttt{" + judge + "} "
        "(baseline pool) on the \\texttt{machine\\_learning} subset, split by "
        "ICLR 2026 acceptance (Accepted $=$ Oral $\\cup$ Poster). "
        "\\textit{Debiased}: per-case median of Reference's $[0,1]$ debiased "
        "win share. \\textit{Win Rate}: per-case median of the AND-of-both win "
        "indicator (Reference wins both forward and reverse; ties score "
        "$0.5$). \\textit{BTD}: Reference's Bradley--Terry--Davidson rating fit "
        "on the bucket's matches alone (opponent-strength adjusted, ELO-like, "
        "centered at $1500$). Brackets are percentile bootstrap $95\\%$ CIs on "
        f"the medians. Per-bucket arena coverage at this snapshot: Accepted "
        f"${n_a_cov}/118$, Rejected ${n_r_cov}/100$.}}\n"
        "\\label{tab:alignment-ml}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{8pt}\n"
        "\\begin{tabular}{lccc}\n"
        "\\toprule\n"
        "\\textbf{Group} & \\textbf{Debiased ($\\uparrow$)} & "
        "\\textbf{Win Rate ($\\uparrow$)} & \\textbf{BTD ($\\uparrow$)} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--judge", default=PRIMARY_JUDGE,
                        help="judge profile (default: seed-2.0-pro)")
    parser.add_argument("--out", default=str(PAPER_TABLES_DIR / "alignment-ml.tex"))
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--update-rows-only", action=argparse.BooleanOptionalAction, default=True,
        help=("Patch only the body rows of an existing target file, preserving "
              "caption / label / column headers (default). Falls back to a full "
              "write when the target file does not exist."),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    bucket_of = _load_bucket_map(args.config)
    n_a = sum(1 for v in bucket_of.values() if v == "Accepted")
    n_r = sum(1 for v in bucket_of.values() if v == "Rejected")
    print(f"Total {DOMAIN} cases with venue: Accept={n_a}, Reject={n_r}")

    matches = load_arena_matches(DOMAIN, args.config, args.judge)
    matches_pool = filter_matches_by_pool(matches, "baseline")

    deb_pc, win_pc = _per_case_metrics(matches_pool)
    deb_buckets = _bucket_split(deb_pc, bucket_of)
    win_buckets = _bucket_split(win_pc, bucket_of)
    deb_summary = _per_case_summary(deb_buckets, n_boot=args.n_boot, seed=args.seed)
    win_summary = _per_case_summary(win_buckets, n_boot=args.n_boot, seed=args.seed)
    bucket_match_lists = _bucket_matches(matches_pool, bucket_of)
    btd_summary = _btd_summary(bucket_match_lists)

    print(f"\n=== {args.judge} ===  (n_matches={len(matches_pool)})")
    print(f"  Per-case Debiased:")
    for b in BUCKETS:
        s = deb_summary[b]
        print(f"    {b:<7s} n={s['n']:>3d}  "
              f"median={s['median']:.3f} CI[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")
    print(f"    MWU one-sided (Accept>Reject): U={deb_summary['mwu']['U']:.0f} "
          f"p={deb_summary['mwu']['p']:.4f}")
    print(f"  Per-case Win Rate:")
    for b in BUCKETS:
        s = win_summary[b]
        print(f"    {b:<7s} n={s['n']:>3d}  "
              f"median={s['median']:.3f} CI[{s['ci_lo']:.3f},{s['ci_hi']:.3f}]")
    print(f"    MWU one-sided (Accept>Reject): U={win_summary['mwu']['U']:.0f} "
          f"p={win_summary['mwu']['p']:.4f}")
    print(f"  Per-bucket BTD:")
    for b in BUCKETS:
        s = btd_summary[b]
        print(f"    {b:<7s} n_matches={s['n_matches']:>5d}  ref BTD={s['ref_btd']:.1f}")
    print(f"    Gap (Accepted - Rejected): {btd_summary['gap']:+.1f}")

    target = Path(args.out)
    if args.update_rows_only and target.exists():
        patch_table_body(target, _build_body(deb_summary, win_summary, btd_summary))
    else:
        write_text_artifact(target,
                             render_table(args.judge, deb_summary, win_summary, btd_summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
