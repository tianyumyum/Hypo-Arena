"""Render per-domain rubric-dimension breakdown sub-tables for the appendix.

Companion to ``paper_score.py``: that script produces one cross-domain table of
mean rubric ``S`` per (model, mode); this one drills into each domain
separately and reports every rubric dimension (grounding, insight,
justification; for real-world domains additionally breadth, distinctness,
utility, recall) so the rubric profile of each entry is visible.

Output: ``tables/score-details-{judge}.tex`` — six ``\\begin{table}`` blocks
in one file, ready to ``\\input`` from an appendix.

Usage:
  uv run python -m scripts.paper_score_details [--judges seed-2.0-pro,...]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import REFERENCE_LABEL, get_domain
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    DOMAIN_FULL_LABEL,
    DOMAIN_SHORT_LABEL,
    ORDERED_DOMAINS,
    PAPER_JUDGES,
    PAPER_MAIN_TABLE_MODELS,
    PAPER_TABLES_DIR,
    display_model_name,
    judge_short_name,
    load_score_leaderboard,
    write_text_artifact,
)


PAIR_DIMS: tuple[str, ...] = ("grounding", "insight", "justification")
SET_DIMS: tuple[str, ...] = ("breadth", "distinctness", "utility")

DIM_HEADERS: dict[str, str] = {
    "grounding": "Ground.",
    "insight": "Insight",
    "justification": "Justif.",
    "breadth": "Breadth",
    "distinctness": "Distinct.",
    "utility": "Utility",
}


@dataclass
class DomainRow:
    """One row of a per-domain rubric-detail sub-table."""

    breakdown: dict[str, float]
    display_name: str
    is_reference: bool
    mode: str                                          # "Baseline" | "Agent" | "" for reference


def _row_label_parse(raw: str) -> tuple[str, str, bool]:
    """Return (profile, mode_label, is_reference) from a raw leaderboard label."""
    if raw == REFERENCE_LABEL:
        return REFERENCE_LABEL, "", True
    if ":" in raw:
        mode_raw, profile = raw.split(":", 1)
        return profile, mode_raw.title(), False
    return raw, "", False


def _collect(domain: str, judge: str, config: str) -> list[DomainRow]:
    """Load full-pool score leaderboard for one domain; restrict to whitelist + reference."""
    eligible = (
        {f"baseline:{p}" for p in PAPER_MAIN_TABLE_MODELS}
        | {f"agent:{p}" for p in PAPER_MAIN_TABLE_MODELS}
        | {REFERENCE_LABEL}
    )
    out: list[DomainRow] = []
    for entry in load_score_leaderboard(domain, judge=judge, config=config, pool="full"):
        if entry.raw_label not in eligible:
            continue
        profile, mode, is_ref = _row_label_parse(entry.raw_label)
        out.append(DomainRow(
            breakdown=dict(entry.breakdown),
            display_name="reference" if is_ref else display_model_name(profile),
            is_reference=is_ref,
            mode=mode,
        ))
    return out


def _fmt(value: float | None, *, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _dims_for_domain(domain: str) -> tuple[str, ...]:
    """Return rubric dims in display order for a domain."""
    cfg = get_domain(domain)
    if cfg.multi_hypothesis:
        return PAIR_DIMS + SET_DIMS
    return PAIR_DIMS


def _render_subtable(*, domain: str, judge: str, rows: list[DomainRow]) -> str:
    """Render one per-domain sub-table; rows sorted by S desc."""
    dims = _dims_for_domain(domain)
    sorted_rows = sorted(rows, key=lambda r: -r.breakdown.get("S", 0.0))

    body_lines: list[str] = []
    for rank, row in enumerate(sorted_rows, start=1):
        cells = [_fmt(row.breakdown.get(d)) for d in dims]
        cells.append(_fmt(row.breakdown.get("S")))
        prefix = (f"\\rowcolor{{colorGT}} {rank} & {row.display_name}"
                  if row.is_reference else
                  f"{rank} & {row.display_name}")
        body_lines.append(
            f"{prefix} & {row.mode} & " + " & ".join(cells) + " \\\\"
        )
    body = "\n".join(body_lines)

    header_cells = " & ".join(f"\\textbf{{{DIM_HEADERS[d]}}}" for d in dims)
    n_numeric_cols = len(dims) + 1                                   # +1 for S
    col_spec = "cll " + "r" * n_numeric_cols                         # rank, model, mode + dims + S

    judge_str = judge_short_name(judge)
    domain_short = DOMAIN_SHORT_LABEL[domain].lower()
    label = f"tab:score-details-{domain_short}-{judge_str}"
    domain_full = DOMAIN_FULL_LABEL[domain]
    caption = (
        f"Per-rubric-dimension breakdown for \\textbf{{{domain_full}}} under "
        f"\\texttt{{{judge}}} (full pool). Rows sorted by Avg $S$ descending; "
        f"all dimensions on the 1--5 scale."
    )
    return (
        "\\begin{table}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n"
        "\\toprule\n"
        f"\\textbf{{\\#}} & \\textbf{{Model}} & \\textbf{{Mode}} & {header_cells} & \\textbf{{Avg $S$}} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def render_file(judge: str, config: str) -> str:
    """Render six sub-tables (one per domain) into a single .tex file body."""
    parts: list[str] = [
        f"% Generated by scripts/paper_score_details.py — judge={judge}",
        "",
    ]
    for domain in ORDERED_DOMAINS:
        rows = _collect(domain, judge, config)
        if not rows:
            print(f"  warning: no rows for domain={domain} judge={judge}; skipping sub-table")
            continue
        parts.append(_render_subtable(domain=domain, judge=judge, rows=rows))
        parts.append("")
    return "\n".join(parts)


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out-dir", default=str(PAPER_TABLES_DIR))
    parser.add_argument("--out-template", default="score-details-{judge}.tex")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    out_dir = Path(args.out_dir)
    for judge in args.judges:
        target = out_dir / args.out_template.format(judge=judge_short_name(judge))
        text = render_file(judge, args.config)
        write_text_artifact(target, text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
