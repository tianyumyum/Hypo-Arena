"""Render Figure of §4.3: arena BTD spread vs rubric S spread per domain.

Horizontal strip-plot layout with three columns:

  * Left  — per-domain strip of arena BTD ratings (one dot per baseline model).
  * Mid   — per-domain compression ratio (arena spread / rubric spread) as a badge.
  * Right — per-domain strip of rubric S scores.

Domains are stacked on the Y-axis with research and real-world groups separated by
a thin rule. Reference is overlaid as a green diamond on the strip it appears in.

Visual style adapted from the paper's pre-Stage-A `plot_arena_vs_score.py` reference.

Usage:
  uv run python -m scripts.paper_arena_vs_score [--judge seed-2.0-pro]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import REAL_WORLD_DOMAINS, REFERENCE_LABEL, RESEARCH_DOMAINS
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    PAPER_IMAGES_DIR,
    PAPER_JUDGES,
    PAPER_MAIN_TABLE_MODELS,
    PRIMARY_JUDGE,
    LeaderboardRow,
    judge_short_name,
    load_arena_leaderboard,
    load_score_leaderboard,
)


# ---- per-domain display labels (two-line for compact y-axis ticks) -----------

DOMAIN_DISPLAY: dict[str, str] = {
    "biomedical_science":   "Biomedical\nScience",
    "machine_learning":     "Machine\nLearning",
    "social_science":       "Social\nScience",
    "financial_analysis":   "Financial\nAnalysis",
    "it_operations":        "IT\nOperations",
    "safety_investigation": "Safety\nInvestigation",
}

# ---- palette (mirrors the reference script) ----------------------------------

C_RESEARCH  = "#4878A8"
C_REALWORLD = "#BF6B56"
C_REFERENCE = "#5B9E5B"
C_ANNOT     = "#777777"
C_RATIO     = "#8B5E3C"
C_SEP       = "#D5D5D5"
BG_COLOR    = "#FBFBFB"


# ---- data collection ---------------------------------------------------------

def _eligible(rows: list[LeaderboardRow], whitelist: set[str]) -> list[LeaderboardRow]:
    keep: list[LeaderboardRow] = []
    for r in rows:
        if r.raw_label == REFERENCE_LABEL:
            keep.append(r)
        elif r.mode == "baseline" and r.profile in whitelist:
            keep.append(r)
    return keep


def _collect(judge: str, config: str, whitelist: set[str]):
    """Return ordered rows: [{label, group, a_vals, s_vals, ref_a, ref_s}, ...]."""
    grouped: list[tuple[int, tuple[str, ...]]] = [
        (0, RESEARCH_DOMAINS),
        (1, REAL_WORLD_DOMAINS),
    ]
    rows: list[dict] = []
    for group_idx, domains in grouped:
        for domain in domains:
            a_rows = _eligible(
                load_arena_leaderboard(domain, judge=judge, config=config, pool="baseline"),
                whitelist,
            )
            s_rows = _eligible(
                load_score_leaderboard(domain, judge=judge, config=config, pool="baseline"),
                whitelist,
            )
            a_vals = [r.rating for r in a_rows if r.raw_label != REFERENCE_LABEL]
            s_vals = [r.rating for r in s_rows if r.raw_label != REFERENCE_LABEL]
            ref_a = next((r.rating for r in a_rows if r.raw_label == REFERENCE_LABEL), None)
            ref_s = next((r.rating for r in s_rows if r.raw_label == REFERENCE_LABEL), None)
            rows.append({
                "label": DOMAIN_DISPLAY[domain],
                "group": group_idx,
                "a_vals": a_vals, "s_vals": s_vals,
                "ref_a": ref_a, "ref_s": ref_s,
            })
    return rows


# ---- rendering ---------------------------------------------------------------

def _render(rows: list[dict], *, out_path: Path, judge: str) -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 8.5,
        "axes.labelsize": 9.5,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    fig = plt.figure(figsize=(7.2, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.13, 1], wspace=0.05)
    ax_arena = fig.add_subplot(gs[0, 0])
    ax_mid   = fig.add_subplot(gs[0, 1])
    ax_score = fig.add_subplot(gs[0, 2])

    n_research = sum(1 for r in rows if r["group"] == 0)
    GAP = 0.6
    y_positions: list[float] = []
    y_labels: list[str] = []
    ratios: list[float] = []
    y_idx = 0.0

    for i, row in enumerate(rows):
        if i == n_research:
            y_idx += GAP

        color = C_RESEARCH if row["group"] == 0 else C_REALWORLD
        a_vals, s_vals = row["a_vals"], row["s_vals"]
        if not a_vals or not s_vals:
            y_positions.append(y_idx)
            y_labels.append(row["label"])
            ratios.append(float("nan"))
            y_idx += 1
            continue
        a_range = max(a_vals) - min(a_vals)
        s_range = max(s_vals) - min(s_vals)
        ratio = a_range / s_range if s_range > 0 else float("inf")
        ratios.append(ratio)

        for ax, vals, ref_val, rng_txt in [
            (ax_arena, a_vals, row["ref_a"], f"{a_range:.0f}"),
            (ax_score, s_vals, row["ref_s"], f"{s_range:.2f}"),
        ]:
            vmin, vmax = min(vals), max(vals)
            ax.plot([vmin, vmax], [y_idx, y_idx],
                    color=color, linewidth=2.0, alpha=0.18,
                    solid_capstyle="round", zorder=1)
            ax.scatter(vals, [y_idx] * len(vals),
                       c=color, s=18, alpha=0.8,
                       edgecolors="white", linewidths=0.4, zorder=2)
            if ref_val is not None:
                ax.scatter([ref_val], [y_idx], c=C_REFERENCE,
                           marker="D", s=28,
                           edgecolors="white", linewidths=0.5, zorder=3)
            ax.annotate(rng_txt, xy=(vmax, y_idx), xytext=(5, 0),
                        textcoords="offset points", fontsize=6, color=C_ANNOT,
                        va="center", ha="left", style="italic")

        y_positions.append(y_idx)
        y_labels.append(row["label"])
        y_idx += 1

    # Invert y so research sits on top.
    for ax in (ax_arena, ax_score):
        ax.set_yticks(y_positions)
        ax.invert_yaxis()

    # Middle column: compression ratio badges.
    ax_mid.set_xlim(0, 1)
    ax_mid.set_ylim(ax_arena.get_ylim())
    ax_mid.axis("off")
    for yp, ratio in zip(y_positions, ratios):
        if ratio != ratio or ratio == float("inf"):    # NaN or inf
            continue
        ax_mid.text(0.5, yp, f"{ratio:.0f}×",
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", color=C_RATIO,
                    bbox=dict(boxstyle="round,pad=0.15",
                              facecolor="#FDF6EC", edgecolor="#E0D0B8",
                              linewidth=0.4))
    ax_mid.text(0.5, ax_mid.get_ylim()[0] + 0.45, "Ratio",
                ha="center", va="top", fontsize=7.5,
                fontweight="bold", color=C_RATIO)

    # Axis styling.
    for ax, title, xlabel in [
        (ax_arena, "(a) Arena Ratings",  "Rating"),
        (ax_score, "(b) Rubric Scores",  "Score"),
    ]:
        ax.set_title(title, fontweight="bold", pad=16, fontsize=10)
        ax.set_xlabel(xlabel, labelpad=4, fontsize=8.5)
        ax.set_facecolor(BG_COLOR)
        ax.grid(axis="x", linestyle=":", alpha=0.35, linewidth=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.4)
        ax.spines["bottom"].set_linewidth(0.4)
        ax.tick_params(axis="y", length=0, pad=5)

    ax_arena.set_yticklabels(y_labels, fontsize=7.5, rotation=25,
                             ha="right", rotation_mode="anchor")
    ax_score.set_yticklabels([])

    # Group separator between research and real-world.
    if n_research > 0 and n_research < len(rows):
        sep_y = y_positions[n_research - 1] + GAP / 2
        for ax in (ax_arena, ax_score):
            ax.axhline(sep_y, color=C_SEP, linewidth=0.6, zorder=0)

    fig.subplots_adjust(top=0.77)

    # Top legend.
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_RESEARCH,
               markersize=4.5, label="Scientific domains"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_REALWORLD,
               markersize=4.5, label="Analytical domains"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_REFERENCE,
               markersize=4, label="Reference"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.85),
               handletextpad=0.3, columnspacing=1.5, fontsize=7.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---- entry -------------------------------------------------------------------

def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out-dir", default=str(PAPER_IMAGES_DIR))
    parser.add_argument("--out-template", default="arena-vs-score-{judge}.pdf")
    parser.add_argument("--no-whitelist", action="store_true")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    out_dir = Path(args.out_dir)
    for judge in args.judges:
        whitelist = set(PAPER_MAIN_TABLE_MODELS) if not args.no_whitelist else set()
        if not whitelist:
            for d in (*RESEARCH_DOMAINS, *REAL_WORLD_DOMAINS):
                for r in load_arena_leaderboard(d, judge=judge, config=args.config, pool="baseline"):
                    if r.mode == "baseline":
                        whitelist.add(r.profile)

        rows = _collect(judge, args.config, whitelist)
        target = out_dir / args.out_template.format(judge=judge_short_name(judge))
        _render(rows, out_path=target, judge=judge)

        print()
        print(f"[{judge}] Per-domain spreads (arena, rubric, ratio):")
        for row in rows:
            a, s = row["a_vals"], row["s_vals"]
            if not a or not s:
                print(f"  {row['label'].replace(chr(10), ' '):<22s}  (no data)")
                continue
            a_full = a + ([row["ref_a"]] if row["ref_a"] is not None else [])
            s_full = s + ([row["ref_s"]] if row["ref_s"] is not None else [])
            a_spread = max(a_full) - min(a_full)
            s_spread = max(s_full) - min(s_full)
            ratio = a_spread / s_spread if s_spread > 0 else float("nan")
            print(f"  {row['label'].replace(chr(10), ' '):<22s}  arena={a_spread:7.1f}  "
                  f"rubric={s_spread:5.2f}  ratio={ratio:6.0f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
