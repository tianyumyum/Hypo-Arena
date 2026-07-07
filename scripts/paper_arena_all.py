"""Render the full-pool arena leaderboard (baseline + agent on one BTD scale).

Companion to ``paper_arena.py`` (which reports the baseline-only main table).
This appendix table puts every (profile, mode) cell on the same full-pool BTD
scale so that mode-level comparisons are read directly off rank order, and
serves as the arena-side reference for the rubric diagnostics in
Table~\\ref{tab:score-seed}.

Output: ``tables/arena-all-{judge}.tex``

Usage:
  uv run python -m scripts.paper_arena_all [--judges seed-2.0-pro,...]
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
    display_model_name,
    judge_short_name,
    load_arena_leaderboard,
    patch_table_body,
    write_text_artifact,
)


@dataclass
class FullRow:
    """One row of the full-pool arena table (one per (profile, mode) cell)."""

    display_name: str
    is_reference: bool
    profile: str
    mode: str                              # "Baseline", "Agent", or "" for Reference
    per_domain_rating: dict[str, float]    # domain → BTD
    per_domain_winrate: dict[str, float]   # domain → WR ∈ [0, 1]

    @property
    def avg_rating(self) -> float | None:
        if not self.per_domain_rating:
            return None
        return sum(self.per_domain_rating.values()) / len(self.per_domain_rating)

    @property
    def avg_winrate(self) -> float | None:
        if not self.per_domain_winrate:
            return None
        return sum(self.per_domain_winrate.values()) / len(self.per_domain_winrate)


def _row_label_parse(raw: str) -> tuple[str, str, bool]:
    """Return (profile, mode_label, is_reference) from a raw leaderboard label."""
    if raw == REFERENCE_LABEL:
        return REFERENCE_LABEL, "", True
    if ":" in raw:
        mode_raw, profile = raw.split(":", 1)
        return profile, mode_raw.title(), False
    return raw, "", False


def _collect(judge: str, config: str) -> list[FullRow]:
    """Load full-pool arena leaderboards; one row per (profile, mode) cell."""
    by_d = {
        d: {r.raw_label: r for r in
            load_arena_leaderboard(d, judge=judge, config=config, pool="full")}
        for d in ORDERED_DOMAINS
    }
    eligible = (
        {f"baseline:{p}" for p in PAPER_MAIN_TABLE_MODELS}
        | {f"agent:{p}" for p in PAPER_MAIN_TABLE_MODELS}
        | {REFERENCE_LABEL}
    )
    rows: list[FullRow] = []
    for raw in sorted(eligible):
        profile, mode, is_ref = _row_label_parse(raw)
        per_rating: dict[str, float] = {}
        per_wr: dict[str, float] = {}
        for d in ORDERED_DOMAINS:
            row = by_d[d].get(raw)
            if row is None:
                continue
            per_rating[d] = row.rating
            wr = row.win_rate
            if wr is not None:
                per_wr[d] = wr
        if not per_rating:
            continue
        rows.append(FullRow(
            display_name="reference" if is_ref else display_model_name(profile),
            is_reference=is_ref,
            profile=profile,
            mode=mode,
            per_domain_rating=per_rating,
            per_domain_winrate=per_wr,
        ))
    return rows


def _fmt_rating(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return "—"
    text = f"{value:.1f}"
    return f"\\textbf{{{text}}}" if bold else text


def _fmt_winrate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}\\%"


def _build_body(rows: list[FullRow]) -> str:
    """Build only the body lines (sorted by avg_rating desc; one (model, mode) per row)."""
    if not rows:
        return ""
    sorted_rows = sorted(
        rows,
        key=lambda r: -(r.avg_rating if r.avg_rating is not None else float("-inf")),
    )
    column_max_per_domain: dict[str, float] = {}
    for d in ORDERED_DOMAINS:
        vals = [r.per_domain_rating[d] for r in sorted_rows if d in r.per_domain_rating]
        if vals:
            column_max_per_domain[d] = max(vals)
    avg_vals = [r.avg_rating for r in sorted_rows if r.avg_rating is not None]
    column_max_avg = max(avg_vals) if avg_vals else 0.0

    body_lines: list[str] = []
    for i, row in enumerate(sorted_rows, start=1):
        per_d_cells = " & ".join(
            _fmt_rating(row.per_domain_rating.get(d),
                        bold=(d in row.per_domain_rating
                              and row.per_domain_rating[d] == column_max_per_domain.get(d)))
            for d in ORDERED_DOMAINS
        )
        avg_cell = _fmt_rating(row.avg_rating,
                               bold=row.avg_rating == column_max_avg)
        wr_cell = _fmt_winrate(row.avg_winrate)
        prefix = (f"\\rowcolor{{colorGT}} {i} & {row.display_name}"
                  if row.is_reference else
                  f"{i} & {row.display_name}")
        body_lines.append(
            f"{prefix} & {row.mode} & {per_d_cells} & {avg_cell} & {wr_cell} \\\\"
        )
    return "\n".join(body_lines)


def render_table(rows: list[FullRow], *, judge: str) -> str:
    """Full table including caption + label."""
    body = _build_body(rows)
    headers = " & ".join(["\\textbf{" + DOMAIN_SHORT_LABEL[d] + "}" for d in ORDERED_DOMAINS])
    caption = (
        f"Full-pool arena leaderboard under \\texttt{{{judge}}}: every "
        f"(model, mode) cell on the same BTD scale across the six domains. "
        f"Rows are sorted by Avg BTD descending; \\colorbox{{colorGT}}{{reference}} "
        f"participates as a single anonymous competitor (no agent counterpart). "
        f"This table is the arena-side counterpart to "
        f"Table~\\ref{{tab:score-{judge_short_name(judge)}}} and the source of the "
        f"\\textit{{Arena \\#}} ordering used there."
    )
    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:arena-all-{judge_short_name(judge)}}}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{cll rrr rrr rr}\n"
        "\\toprule\n"
        " & & & \\multicolumn{3}{c}{\\textbf{Research}} & \\multicolumn{3}{c}{\\textbf{Real-World}} & & \\\\\n"
        "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}\n"
        f"\\textbf{{\\#}} & \\textbf{{Model}} & \\textbf{{Mode}} & {headers} & "
        f"\\textbf{{Avg}} & \\textbf{{WR}} \\\\\n"
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
    print(f"Full-pool arena (judge={judge}):")
    sorted_rows = sorted(
        rows,
        key=lambda r: -(r.avg_rating if r.avg_rating is not None else float("-inf")),
    )
    for i, row in enumerate(sorted_rows, start=1):
        avg = "—" if row.avg_rating is None else f"{row.avg_rating:7.1f}"
        wr = "—" if row.avg_winrate is None else f"{row.avg_winrate * 100:5.1f}%"
        print(f"  {i:>2}.  {row.display_name:<22s} {row.mode:<9s}  avg={avg}  WR={wr}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out-dir", default=str(PAPER_TABLES_DIR))
    parser.add_argument("--out-template", default="arena-all-{judge}.tex")
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
