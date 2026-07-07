"""Render §4.5 Table (`tables/recall.tex`): per-(model, mode) recall + arena rank.

Reads `score.jsonl` from real-world domains (recall is only computed where
`domain.multi_hypothesis=True` and the score judge was run with `--with-recall`),
aggregates per (model_label) into mean hits / total / recall ± std, drops rows with
n < 50, sorts by recall desc, and assigns Arena Rank from the full-pool BTD averaged
across all 6 domains.

Reports Spearman ρ between Recall Rank and Arena Rank across the displayed rows.

Usage:
  uv run python -m scripts.paper_recall [--judge mimo-v2-pro]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import RESEARCH_DOMAINS, REAL_WORLD_DOMAINS, REFERENCE_LABEL
from basics.io import load_score_records
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    ORDERED_DOMAINS,
    PAPER_JUDGES,
    PAPER_MAIN_TABLE_MODELS,
    PAPER_TABLES_DIR,
    PRIMARY_JUDGE,
    display_model_name,
    judge_short_name,
    load_arena_leaderboard,
    patch_table_body,
    rows_by_profile,
    split_mode_label,
)
from scripts._paper_stats import mean_std, spearman_rho


_DOMAIN_SETS: dict[str, tuple[str, ...]] = {
    "all": ORDERED_DOMAINS,
    "realworld": REAL_WORLD_DOMAINS,
    "research": RESEARCH_DOMAINS,
}


@dataclass
class RecallRow:
    label: str                    # raw label e.g. "agent:kimi-k2.6-thinking"
    mode: str                     # "baseline" / "agent" / "reference"
    profile: str
    n_cases: int
    mean_hits: float
    mean_total: float
    mean_recall: float
    std_recall: float
    arena_rating: float           # cross-domain mean BTD (full pool)


def _aggregate_recall(judge: str, config: str) -> dict[str, dict]:
    """label → {hits, total, recalls, n_cases}.

    Reference is excluded: its recall is mechanically 1.0 because the score judge
    compares it against itself (Reference-as-Candidate uses reference_submission
    which equals BenchmarkCase.hypotheses).
    """
    out: dict[str, dict] = {}
    for domain in REAL_WORLD_DOMAINS:
        for r in load_score_records(domain, config, judge):
            if r.recall is None or r.recall.total == 0:
                continue
            if r.model == REFERENCE_LABEL:
                continue
            entry = out.setdefault(r.model, {"hits": [], "total": [], "ratio": []})
            entry["hits"].append(r.recall.hits)
            entry["total"].append(r.recall.total)
            entry["ratio"].append(r.recall.ratio)
    return out


def _arena_rating_per_label(
    judge: str, config: str, *, domains: tuple[str, ...] = ORDERED_DOMAINS,
) -> dict[str, float]:
    """label → mean BTD rating across requested domains, full pool."""
    accum: dict[str, list[float]] = {}
    for domain in domains:
        for r in load_arena_leaderboard(domain, judge=judge, config=config, pool="full"):
            accum.setdefault(r.raw_label, []).append(r.rating)
    return {label: sum(vals) / len(vals) for label, vals in accum.items() if vals}


def _build_rows(
    judge: str,
    config: str,
    *,
    min_cases: int,
    arena_domains: tuple[str, ...] = ORDERED_DOMAINS,
) -> list[RecallRow]:
    raw = _aggregate_recall(judge, config)
    arena_by_label = _arena_rating_per_label(judge, config, domains=arena_domains)

    rows: list[RecallRow] = []
    for label, payload in raw.items():
        n_cases = len(payload["ratio"])
        if n_cases < min_cases:
            continue
        if label not in arena_by_label:
            continue                     # no arena data for this label
        mode, profile = split_mode_label(label)
        if profile not in PAPER_MAIN_TABLE_MODELS:
            continue                     # restrict to the §4.2 main-table model set
        mean_recall, std_recall = mean_std(payload["ratio"])
        rows.append(RecallRow(
            label=label,
            mode=mode,
            profile=profile,
            n_cases=n_cases,
            mean_hits=sum(payload["hits"]) / n_cases,
            mean_total=sum(payload["total"]) / n_cases,
            mean_recall=mean_recall,
            std_recall=std_recall,
            arena_rating=arena_by_label[label],
        ))
    rows.sort(key=lambda r: -r.mean_recall)
    return rows


def _setting_label(mode: str) -> str:
    return {"baseline": "Baseline", "agent": "Agent", "reference": "Reference"}.get(mode, mode)


def _display_name(row: RecallRow) -> str:
    if row.mode == "reference":
        return "Reference"
    return display_model_name(row.profile)


def _build_body(rows: list[RecallRow]) -> str:
    """Build only the body region (data rows + inner \\midrule + Spearman footer).

    Used by `render_table` (full output) and `--update-rows-only` (in-place body
    patch that preserves caption / label / column headers).
    """
    if not rows:
        return ""

    # Arena rank assigned within displayed rows (1 = highest arena rating).
    sorted_by_arena = sorted(range(len(rows)), key=lambda i: -rows[i].arena_rating)
    arena_rank = [0] * len(rows)
    for rank, idx in enumerate(sorted_by_arena, start=1):
        arena_rank[idx] = rank

    recall_ranks = list(range(1, len(rows) + 1))
    rho = spearman_rho([float(r) for r in recall_ranks], [float(a) for a in arena_rank])

    body_lines: list[str] = []
    for i, row in enumerate(rows):
        delta = arena_rank[i] - (i + 1)
        delta_str = f"$\\Delta${delta:+d}" if delta else "$\\Delta$0"
        recall_cell = (f"{row.mean_recall:.3f}".lstrip("0") +
                       " $\\pm$ " + f"{row.std_recall:.3f}".lstrip("0"))
        bold_top = " \\textbf{" if i == 0 else " "
        bold_close = "}" if i == 0 else ""
        recall_text = bold_top + recall_cell + bold_close
        body_lines.append(
            f"{i+1} & {_display_name(row)} & {_setting_label(row.mode)} & "
            f"{row.mean_hits:.2f} & {row.mean_total:.2f} & "
            f"{recall_text} & {arena_rank[i]}~{{\\scriptsize ({delta_str})}} \\\\"
        )

    body_lines.append("\\midrule")
    body_lines.append(
        f"\\multicolumn{{7}}{{l}}{{\\footnotesize Spearman $\\rho = {rho:.3f}$ "
        f"between Recall rank and Arena rank ($n={len(rows)}$).}} \\\\"
    )
    return "\n".join(body_lines)


def render_table(rows: list[RecallRow], *, judge: str, arena_domain_set: str) -> str:
    """Render the full recall table including caption + label."""
    if not rows:
        return "% No eligible rows for the recall table.\n"

    body = _build_body(rows)
    # Recompute rho for caption (also computed inside _build_body; both use the
    # same recall/arena rank vectors so the value is identical).
    sorted_by_arena = sorted(range(len(rows)), key=lambda i: -rows[i].arena_rating)
    arena_rank = [0] * len(rows)
    for rank, idx in enumerate(sorted_by_arena, start=1):
        arena_rank[idx] = rank
    rho = spearman_rho(
        [float(r) for r in range(1, len(rows) + 1)],
        [float(a) for a in arena_rank],
    )

    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Reference recall statistics per model, with arena rank alignment. "
        "\\textit{Hits} = mean number of reference causal factors recovered per case; "
        "\\textit{Total} = mean reference factors available per case; "
        "\\textit{Recall} = per-case mean of Hits/Total. \\textit{Arena Rank} is derived from the full-pool "
        f"BTD averaged across {arena_domain_set} domains, restricted to the rows shown. "
        f"Only configurations with $n \\geq 50$ recall-evaluated instances are included; "
        f"rows are sorted by Recall (descending). The Recall ranking and Arena ranking show "
        f"agreement (Spearman $\\rho = {rho:.3f}$, $n={len(rows)}$). "
        f"Judge: \\texttt{{{judge}}}.}}\n"
        f"\\label{{tab:recall-{judge_short_name(judge)}}}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n"
        "\\begin{tabular}{clcrrcr}\n"
        "\\toprule\n"
        "\\textbf{Recall Rank} & \\textbf{Model} & \\textbf{Mode} "
        "& \\textbf{Hits} & \\textbf{Total} "
        "& \\textbf{Recall $\\pm$ $\\sigma$} & \\textbf{Arena Rank} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out-dir", default=str(PAPER_TABLES_DIR))
    parser.add_argument("--out-template", default="recall-{judge}.tex")
    parser.add_argument("--min-cases", type=int, default=50,
                        help="Drop (model, mode) entries with fewer recall-evaluated cases.")
    parser.add_argument(
        "--arena-domains", choices=tuple(_DOMAIN_SETS), default="realworld",
        help=("Which domains feed the arena BTD that drives Arena Rank. 'realworld' "
              "(default) restricts arena to the same 3 domains where recall is measured "
              "(apples-to-apples scope, methodologically correct for this comparison); "
              "'all' matches the §4.2 main leaderboard; 'research' for completeness."),
    )
    parser.add_argument(
        "--update-rows-only", action=argparse.BooleanOptionalAction, default=True,
        help=("Patch only the body rows of an existing target file, preserving "
              "caption / label / column headers (default). Falls back to a full "
              "write when the target file does not exist."),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    arena_domains = _DOMAIN_SETS[args.arena_domains]
    out_dir = Path(args.out_dir)

    # Build per-judge eligible row lists, then keep only the (profile, mode)
    # cells that pass the n_cases threshold under EVERY judge so all rendered
    # tables expose the same row set (apples-to-apples cross-judge comparison).
    per_judge_rows: dict[str, list] = {
        judge: _build_rows(
            judge, args.config,
            min_cases=args.min_cases, arena_domains=arena_domains,
        )
        for judge in args.judges
    }
    common_keys: set[tuple[str, str]] | None = None
    for rows in per_judge_rows.values():
        keys = {(r.profile, r.mode) for r in rows}
        common_keys = keys if common_keys is None else common_keys & keys
    common_keys = common_keys or set()

    for judge in args.judges:
        rows = [r for r in per_judge_rows[judge]
                if (r.profile, r.mode) in common_keys]
        target = out_dir / args.out_template.format(judge=judge_short_name(judge))
        if args.update_rows_only and target.exists():
            patch_table_body(target, _build_body(rows))
        else:
            text = render_table(rows, judge=judge, arena_domain_set=args.arena_domains)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
            print(f"wrote {target}  ({len(text):,} bytes)")

        print()
        print(f"[{judge}] Rows after cross-judge intersection: {len(rows)}")
        if rows:
            sorted_by_arena = sorted(range(len(rows)), key=lambda i: -rows[i].arena_rating)
            arena_rank = [0] * len(rows)
            for rank, idx in enumerate(sorted_by_arena, start=1):
                arena_rank[idx] = rank
            for i, row in enumerate(rows):
                delta = arena_rank[i] - (i + 1)
                sign = "+" if delta > 0 else ("-" if delta < 0 else " ")
                print(f"  {i+1:>2}.  {_display_name(row):<22s} {_setting_label(row.mode):<9s}  "
                      f"n={row.n_cases:>3d}  recall={row.mean_recall:.3f}±{row.std_recall:.3f}  "
                      f"arena_rank={arena_rank[i]:>2d} (Δ{sign}{abs(delta)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
