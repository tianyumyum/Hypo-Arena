"""Render the dataset domain distribution pie chart for §2.

Per-domain case counts are computed live from `artifacts/{domain}/cases/*.jsonl`
under the construction profile, so the figure stays in sync with the data on disk
without any hardcoded numbers.

Output: ``images/domain-distribution.pdf``

Usage:
  uv run python -m scripts.paper_domain_distribution
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics.io import load_cases
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    DOMAIN_FULL_LABEL,
    ORDERED_DOMAINS,
    PAPER_IMAGES_DIR,
)


# Two-line display labels (line break after the first word for compact slices).
DOMAIN_TWO_LINE: dict[str, str] = {
    "biomedical_science":   "Biomedical\nScience",
    "machine_learning":     "Machine\nLearning",
    "social_science":       "Social\nScience",
    "financial_analysis":   "Financial\nAnalysis",
    "it_operations":        "IT\nOperations",
    "safety_investigation": "Safety\nInvestigation",
}

# Research domains get a blue gradient, real-world get a warm rose gradient.
RESEARCH_DOMAINS = ("biomedical_science", "machine_learning", "social_science")
REALWORLD_DOMAINS = ("financial_analysis", "it_operations", "safety_investigation")
RESEARCH_COLORS  = ["#a5bbe3", "#849cc7", "#6780ab"]
REALWORLD_COLORS = ["#f7e1e1", "#dbb8b8", "#bf9090"]


# Visual order for the pie chart (IT and Safety swapped vs global ORDERED_DOMAINS
# to put Safety's tapered wedge next to Social Science's wider one, balancing
# the circle).
PIE_DOMAIN_ORDER: tuple[str, ...] = (
    "biomedical_science",
    "machine_learning",
    "social_science",
    "financial_analysis",
    "safety_investigation",
    "it_operations",
)


def _domain_counts(config: str) -> list[tuple[str, int]]:
    """Return [(domain, n_cases), ...] in `PIE_DOMAIN_ORDER`."""
    out: list[tuple[str, int]] = []
    for domain in PIE_DOMAIN_ORDER:
        cases = load_cases(domain, config)
        out.append((domain, len(cases)))
    return out


def _render(counts: list[tuple[str, int]], out_path: Path) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

    sizes = [n for _, n in counts]
    labels = [DOMAIN_TWO_LINE[d] for d, _ in counts]
    is_realworld = [d in REALWORLD_DOMAINS for d, _ in counts]
    colors = (RESEARCH_COLORS[: sum(1 for d, _ in counts if d in RESEARCH_DOMAINS)]
              + REALWORLD_COLORS[: sum(1 for d, _ in counts if d in REALWORLD_DOMAINS)])

    fig, ax = plt.subplots(figsize=(3.8, 3.2))
    wedges, _ = ax.pie(
        sizes,
        labels=None,
        colors=colors,
        startangle=90,
        counterclock=False,
        wedgeprops=dict(edgecolor="white", linewidth=1.8),
    )

    # Uniform radial placement and font size across all slices.
    r_adj = 0.68
    fontsize = 6.0
    for wedge, label, size, realworld in zip(wedges, labels, sizes, is_realworld):
        angle = (wedge.theta2 + wedge.theta1) / 2
        x = r_adj * np.cos(np.radians(angle))
        y = r_adj * np.sin(np.radians(angle))
        txt_color = "#333333" if realworld else "white"
        ax.text(
            x, y, f"{label}\n({size})",
            ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=txt_color,
            linespacing=1.15,
        )

    ax.set_aspect("equal")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out", default=str(PAPER_IMAGES_DIR / "domain-distribution.pdf"))
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    counts = _domain_counts(args.config)
    total = sum(n for _, n in counts)
    print()
    print(f"Per-domain case counts (config={args.config}):")
    for domain, n in counts:
        print(f"  {DOMAIN_FULL_LABEL[domain]:<22s} {n:>4d}")
    print(f"  {'Total':<22s} {total:>4d}")
    print()

    _render(counts, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
