"""Render §2's dataset statistics table from artifacts/{domain}/cases/*.jsonl.

Computes live per-domain counts (cases, hypothesis-evidence pairs, average set
size, unique categories) and patches the body of ``tables/dataset-statistics.tex``
so the table stays in sync with the data on disk. Static columns (domain label,
Primary Source citation) and the caption are preserved.

Usage:
  uv run python -m scripts.paper_dataset
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics.io import load_cases
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    DOMAIN_FULL_LABEL,
    PAPER_TABLES_DIR,
    patch_table_body,
)


RESEARCH_DOMAINS = ("biomedical_science", "machine_learning", "social_science")
REALWORLD_DOMAINS = ("financial_analysis", "it_operations", "safety_investigation")

PRIMARY_SOURCE: dict[str, str] = {
    "biomedical_science":   "Journal Articles",
    "machine_learning":     "Conference Papers",
    "social_science":       "Journal Articles",
    "financial_analysis":   "SEC 10-Q + Analyst Reports",
    "it_operations":        "Post-mortems Blogs",
    "safety_investigation": "CSB \\& NTSB Reports",
}


def _format_int(n: int) -> str:
    """Render large integers with LaTeX thousands separator: 1192 → 1{,}192."""
    return f"{n:,}".replace(",", "{,}")


def _distinct_categories(case) -> int:
    """How many distinct categories are spanned by this case's H-E pairs (None counts as one)."""
    return len({h.category for h in case.hypotheses})


def _collect(config: str) -> dict[str, dict]:
    """Per-domain: n_cases, total H-E pair count, and per-case distinct-category counts."""
    out: dict[str, dict] = {}
    for domain in RESEARCH_DOMAINS + REALWORLD_DOMAINS:
        cases = load_cases(domain, config)
        per_case_cats = [_distinct_categories(c) for c in cases]
        out[domain] = {
            "n_cases": len(cases),
            "n_he": sum(len(case.hypotheses) for case in cases),
            "sum_distinct_cats": sum(per_case_cats),
        }
    return out


def _domain_row(domain: str, stats: dict, *, show_avg_cat: bool) -> str:
    """One per-domain line inside the body."""
    if show_avg_cat:
        avg_cat = stats["sum_distinct_cats"] / stats["n_cases"] if stats["n_cases"] else 0.0
        avg_cell = f"{avg_cat:.2f}"
    else:
        avg_cell = ""
    return (
        f"\\quad {DOMAIN_FULL_LABEL[domain]} & {PRIMARY_SOURCE[domain]} & "
        f"{_format_int(stats['n_cases'])} & {_format_int(stats['n_he'])} & "
        f"{avg_cell} \\\\"
    )


def _build_body(per_domain: dict[str, dict]) -> str:
    lines: list[str] = []

    lines.append("\\textit{Research} & & & & \\\\")
    for d in RESEARCH_DOMAINS:
        lines.append(_domain_row(d, per_domain[d], show_avg_cat=False))

    lines.append("\\midrule")
    lines.append("\\textit{Real-World} & & & & \\\\")
    for d in REALWORLD_DOMAINS:
        lines.append(_domain_row(d, per_domain[d], show_avg_cat=True))

    total_cases = sum(s["n_cases"] for s in per_domain.values())
    total_he = sum(s["n_he"] for s in per_domain.values())
    rw_cases = sum(per_domain[d]["n_cases"] for d in REALWORLD_DOMAINS)
    rw_distinct = sum(per_domain[d]["sum_distinct_cats"] for d in REALWORLD_DOMAINS)
    total_avg_cat = rw_distinct / rw_cases if rw_cases else 0.0
    lines.append("\\midrule")
    lines.append(
        f"\\textbf{{Total}} & & \\textbf{{{_format_int(total_cases)}}} & "
        f"\\textbf{{{_format_int(total_he)}}} & \\textbf{{{total_avg_cat:.2f}}} \\\\"
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out", default=str(PAPER_TABLES_DIR / "dataset-statistics.tex"))
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    stats = _collect(args.config)
    body = _build_body(stats)
    patch_table_body(Path(args.out), body)

    print()
    print(f"{'Domain':<22s}  {'Cases':>6s}  {'H-E':>6s}  {'Avg-Cat':>7s}")
    for d in RESEARCH_DOMAINS:
        s = stats[d]
        print(f"  {DOMAIN_FULL_LABEL[d]:<20s}  {s['n_cases']:>6d}  "
              f"{s['n_he']:>6d}  {'—':>7s}")
    for d in REALWORLD_DOMAINS:
        s = stats[d]
        avg_cat = s["sum_distinct_cats"] / s["n_cases"] if s["n_cases"] else 0.0
        print(f"  {DOMAIN_FULL_LABEL[d]:<20s}  {s['n_cases']:>6d}  "
              f"{s['n_he']:>6d}  {avg_cat:>7.2f}")
    total_cases = sum(s["n_cases"] for s in stats.values())
    total_he = sum(s["n_he"] for s in stats.values())
    rw_cases = sum(stats[d]["n_cases"] for d in REALWORLD_DOMAINS)
    rw_distinct = sum(stats[d]["sum_distinct_cats"] for d in REALWORLD_DOMAINS)
    total_avg_cat = rw_distinct / rw_cases if rw_cases else 0.0
    print(f"  {'Total':<20s}  {total_cases:>6d}  {total_he:>6d}  {total_avg_cat:>7.2f}  "
          f"(real-world only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
