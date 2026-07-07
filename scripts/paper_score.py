"""Render the rubric-scoring diagnostics table for the appendix.

This is the rubric-side companion to ``paper_arena.py`` and exists to substantiate
§4.3's "ranking instability" cascade rather than to provide a competing leaderboard.
Rows are sorted by **arena rank** (not rubric S) so the table reads as a
visual diff: scan down the arena ordering and watch where rubric-rank Δ jumps.

Output: ``tables/score-{judge}.tex``

Usage:
  uv run python -m scripts.paper_score [--judges seed-2.0-pro,...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import REFERENCE_LABEL
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    DOMAIN_SHORT_LABEL,
    ORDERED_DOMAINS,
    PAPER_JUDGES,
    PAPER_MAIN_TABLE_MODELS,
    PAPER_TABLES_DIR,
    LeaderboardRow,
    display_model_name,
    judge_short_name,
    load_arena_leaderboard,
    load_score_leaderboard,
    patch_table_body,
    rows_by_profile,
    write_text_artifact,
)


@dataclass
class ModelRow:
    """One row of the rubric-score diagnostic table (one per (profile, mode) cell)."""

    display_name: str
    is_reference: bool
    profile: str
    mode: str                             # "Baseline", "Agent", or "" for Reference
    arena_avg: float | None
    score_per_domain: dict[str, float]   # domain → mean rubric S

    @property
    def score_avg(self) -> float | None:
        if not self.score_per_domain:
            return None
        return sum(self.score_per_domain.values()) / len(self.score_per_domain)


def _row_label_parse(raw: str) -> tuple[str, str, bool]:
    """Return (profile, mode_label, is_reference) from a raw leaderboard label."""
    if raw == REFERENCE_LABEL:
        return REFERENCE_LABEL, "", True
    if ":" in raw:
        mode_raw, profile = raw.split(":", 1)
        return profile, mode_raw.title(), False
    return raw, "", False


def _collect(judge: str, config: str) -> list[ModelRow]:
    """Load full-pool arena + score leaderboards; one row per (profile, mode) cell."""
    arena_by_d = {
        d: {r.raw_label: r.rating for r in
            load_arena_leaderboard(d, judge=judge, config=config, pool="full")}
        for d in ORDERED_DOMAINS
    }
    score_by_d = {
        d: {r.raw_label: r.rating for r in
            load_score_leaderboard(d, judge=judge, config=config, pool="full")}
        for d in ORDERED_DOMAINS
    }

    eligible = (
        {f"baseline:{p}" for p in PAPER_MAIN_TABLE_MODELS}
        | {f"agent:{p}" for p in PAPER_MAIN_TABLE_MODELS}
        | {REFERENCE_LABEL}
    )
    rows: list[ModelRow] = []
    for raw in sorted(eligible):
        profile, mode, is_ref = _row_label_parse(raw)
        arena_vals = [arena_by_d[d].get(raw) for d in ORDERED_DOMAINS]
        arena_vals = [v for v in arena_vals if v is not None]
        score_per_d = {d: score_by_d[d][raw]
                       for d in ORDERED_DOMAINS if raw in score_by_d[d]}
        if not arena_vals or not score_per_d:
            continue
        rows.append(ModelRow(
            display_name="reference" if is_ref else display_model_name(profile),
            is_reference=is_ref,
            profile=profile,
            mode=mode,
            arena_avg=sum(arena_vals) / len(arena_vals),
            score_per_domain=score_per_d,
        ))
    return rows


def _fmt_score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def _build_body(rows: list[ModelRow]) -> str:
    """Build only the body lines (sorted by arena_avg desc, with Δrank vs rubric)."""
    if not rows:
        return ""
    arena_sorted_idx = sorted(range(len(rows)), key=lambda i: -(rows[i].arena_avg or 0))
    arena_rank = [0] * len(rows)
    for r, idx in enumerate(arena_sorted_idx, start=1):
        arena_rank[idx] = r

    rubric_sorted_idx = sorted(range(len(rows)),
                               key=lambda i: -(rows[i].score_avg or 0))
    rubric_rank = [0] * len(rows)
    for r, idx in enumerate(rubric_sorted_idx, start=1):
        rubric_rank[idx] = r

    body_lines: list[str] = []
    for i in arena_sorted_idx:
        row = rows[i]
        ar, rr = arena_rank[i], rubric_rank[i]
        delta = rr - ar
        delta_str = (f"$\\Delta${delta:+d}" if delta else "$\\Delta$0")
        per_d_cells = " & ".join(
            _fmt_score(row.score_per_domain.get(d)) for d in ORDERED_DOMAINS
        )
        avg_cell = _fmt_score(row.score_avg)
        rubric_cell = f"{rr}~{{\\scriptsize ({delta_str})}}"
        prefix = (f"\\rowcolor{{colorGT}} {ar} & {row.display_name}"
                  if row.is_reference else
                  f"{ar} & {row.display_name}")
        body_lines.append(
            f"{prefix} & {row.mode} & {per_d_cells} & {avg_cell} & {rubric_cell} \\\\"
        )
    return "\n".join(body_lines)


def render_table(rows: list[ModelRow], *, judge: str) -> str:
    """Full table including caption + label."""
    body = _build_body(rows)
    headers = " & ".join(["\\textbf{" + DOMAIN_SHORT_LABEL[d] + "}" for d in ORDERED_DOMAINS])
    caption = (
        f"Rubric scoring diagnostics under \\texttt{{{judge}}}: per-(model, mode) "
        f"mean rubric $S$ on the 1--5 scale, complementing "
        f"Figure~\\ref{{fig:arena-vs-score}} of Section~\\ref{{subsec:arena-vs-score}} and "
        f"Table~\\ref{{tab:arena-all-seed}}. Rows are sorted by arena rank "
        f"(Arena \\#) on the same full-pool scale as that table; "
        f"\\textit{{Rubric Rank}} is the position when the same cells are re-ranked "
        f"by Avg $S$ (descending), with $\\Delta = $ Rubric Rank $-$ Arena Rank. "
        f"The narrow Avg $S$ spread and frequent non-zero $\\Delta$ instantiate the "
        f"compression and ranking-instability claims of "
        f"Section~\\ref{{subsec:arena-vs-score}}."
    )
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:score-{judge_short_name(judge)}}}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{cll rrr rrr rc}\n"
        "\\toprule\n"
        " & & & \\multicolumn{3}{c}{\\textbf{Research}} & \\multicolumn{3}{c}{\\textbf{Real-World}} & & \\\\\n"
        "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}\n"
        f"\\textbf{{Arena \\#}} & \\textbf{{Model}} & \\textbf{{Mode}} & {headers} & "
        f"\\textbf{{Avg $S$}} & \\textbf{{Rubric Rank ($\\Delta$)}} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _render_one_judge(judge: str, config: str, target: Path,
                      *, update_rows_only: bool) -> None:
    rows = _collect(judge, config)
    if update_rows_only and target.exists():
        patch_table_body(target, _build_body(rows))
    else:
        write_text_artifact(target, render_table(rows, judge=judge))

    print()
    print(f"Rubric diagnostics (judge={judge}):")
    arena_sorted = sorted(rows, key=lambda r: -(r.arena_avg or 0))
    for i, row in enumerate(arena_sorted, start=1):
        avg = "—" if row.score_avg is None else f"{row.score_avg:.3f}"
        print(f"  {i:>2}.  {row.display_name:<28s}  Avg S={avg}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out-dir", default=str(PAPER_TABLES_DIR))
    parser.add_argument("--out-template", default="score-{judge}.tex")
    parser.add_argument("--update-rows-only", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="Patch only body rows of an existing target file "
                             "(default). Pass --no-update-rows-only to overwrite "
                             "the entire file (caption included).")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    out_dir = Path(args.out_dir)
    for judge in args.judges:
        target = out_dir / args.out_template.format(judge=judge_short_name(judge))
        _render_one_judge(judge, args.config, target,
                          update_rows_only=args.update_rows_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
