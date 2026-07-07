"""Render Figure of §4.5: arena rating vs Reference recall per (model, mode).

Single-panel scatter that visualises both §4.5 findings simultaneously:

  * Finding 1 (convergence) — points form a positive arena/recall trend; Spearman
    rank correlation reported in-figure.
  * Finding 2 (diagnostic disagreement) — outliers fall off the trend. Anchor
    models are picked **data-driven** from the largest +Δrank (recall over-rates
    arena: high recall, low arena) and largest −Δrank (arena over-rates recall:
    high arena, low recall) per run. Within-model baseline-agent pairs are
    connected by thin grey segments to show how the two metrics co-move (or
    fail to) under mode switching.

Re-uses the data backbone of ``paper_recall.py`` to guarantee numerical
consistency with Table~\\ref{tab:recall}.

Usage:
  uv run python -m scripts.paper_arena_vs_recall [--judge seed-2.0-pro]
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

from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    PAPER_IMAGES_DIR,
    PAPER_JUDGES,
    PRIMARY_JUDGE,
    display_model_name,
    judge_short_name,
)
from scripts._paper_stats import spearman_rho
from scripts.paper_recall import _DOMAIN_SETS, _build_rows


# ---- palette (paired with §4.4 figure: Agent lighter, Baseline darker) -------

C_AGENT     = "#a0b7e1"
C_BASELINE  = "#5a7ab0"
C_PAIR_LINE = "#bcc4cc"
C_HIGHLIGHT = "#bf6b56"     # warm coral — headline outlier (high recall, low arena)
C_COUNTER   = "#5b9e5b"     # green — counter-example (top arena, moderate recall)
C_TREND     = "#999999"
BG_COLOR    = "#FBFBFB"


# ---- annotation policy ------------------------------------------------------

# Anchors are picked data-driven by `_pick_anchors`; see docstring.


def _pick_anchors(rows) -> tuple[str, str]:
    """Pick (headline_outlier, counter_example) display names data-driven.

    headline = (model, mode) with max +Δrank (arena_rank − recall_rank);
               recall over-rates arena: high recall but low arena rank.
    counter  = (model, mode) with max −Δrank;
               arena over-rates recall: high arena rank but low recall.
    """
    sorted_by_arena = sorted(range(len(rows)), key=lambda i: -rows[i].arena_rating)
    arena_rank = [0] * len(rows)
    for rank, idx in enumerate(sorted_by_arena, start=1):
        arena_rank[idx] = rank

    deltas: list[tuple[str, int]] = []
    for i, r in enumerate(rows):
        delta = arena_rank[i] - (i + 1)
        deltas.append((display_model_name(r.profile), delta))

    deltas_sorted = sorted(deltas, key=lambda x: x[1])
    counter_name = deltas_sorted[0][0]      # most negative Δ
    headline_name = deltas_sorted[-1][0]    # most positive Δ
    return headline_name, counter_name


def _render(rows, *, out_path: Path, judge: str, arena_domain_set: str) -> None:
    # Drop unpaired (model, mode) cells: keep only models with both baseline AND agent
    # passing the n_cases threshold, so every plotted point participates in a pair.
    modes_per_model: dict[str, set[str]] = {}
    for r in rows:
        modes_per_model.setdefault(display_model_name(r.profile), set()).add(r.mode)
    paired = {m for m, modes in modes_per_model.items() if {"baseline", "agent"} <= modes}
    dropped = sorted(set(modes_per_model) - paired)
    if dropped:
        print(f"  dropping unpaired models from figure: {dropped}")
    rows = [r for r in rows if display_model_name(r.profile) in paired]

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "figure.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.08,
    })

    # Pick narrative anchors from the data (largest +Δ and largest -Δ rank gaps).
    headline_name, counter_name = _pick_anchors(rows)
    print(f"  data-driven anchors: headline_outlier={headline_name!r}, "
          f"counter_example={counter_name!r}")

    def _is_outlier(profile: str) -> bool:
        return profile == headline_name

    def _is_counter(profile: str) -> bool:
        return profile == counter_name

    # Group rows per (display) profile so we can connect Baseline–Agent pairs.
    by_profile: dict[str, dict[str, dict]] = {}
    for r in rows:
        name = display_model_name(r.profile)
        slot = "agent" if r.mode == "agent" else "baseline"
        by_profile.setdefault(name, {})[slot] = {
            "recall": r.mean_recall, "arena": r.arena_rating,
            "is_outlier": _is_outlier(name),
        }

    fig, ax = plt.subplots(figsize=(5.6, 4.0), dpi=300)
    ax.set_facecolor(BG_COLOR)

    # 1) Within-model pair lines (drawn first → underneath markers).
    for name, by_mode in by_profile.items():
        b = by_mode.get("baseline")
        a = by_mode.get("agent")
        if b is None or a is None:
            continue
        ax.plot([b["recall"], a["recall"]], [b["arena"], a["arena"]],
                color=C_PAIR_LINE, linewidth=0.8, alpha=0.7, zorder=1)

    # 2) Markers — Agent vs Baseline distinguished by colour AND shape;
    #    headline outlier highlighted in coral, counter-example in green.
    for r in rows:
        name = display_model_name(r.profile)
        marker = "^" if r.mode == "agent" else "o"
        face = C_AGENT if r.mode == "agent" else C_BASELINE
        if _is_outlier(name):
            edge, edge_w, size = C_HIGHLIGHT, 1.4, 78
        elif _is_counter(name):
            edge, edge_w, size = C_COUNTER, 1.4, 78
        else:
            edge, edge_w, size = "white", 0.5, 42
        ax.scatter(r.mean_recall, r.arena_rating,
                   marker=marker, s=size, c=face,
                   edgecolors=edge, linewidths=edge_w, zorder=3)

    # 3) Linear trend — fit ONLY on the cluster (exclude headline outlier so the
    #    trend reflects the bulk; otherwise the outlier drags the line toward itself).
    cluster = [r for r in rows if not _is_outlier(display_model_name(r.profile))]
    xs_c = [r.mean_recall for r in cluster]
    ys_c = [r.arena_rating for r in cluster]
    if len(xs_c) >= 2:
        n = len(xs_c)
        mx = sum(xs_c) / n
        my = sum(ys_c) / n
        denom = sum((x - mx) ** 2 for x in xs_c)
        if denom > 0:
            slope = sum((x - mx) * (y - my) for x, y in zip(xs_c, ys_c)) / denom
            intercept = my - slope * mx
            xr = (min(xs_c), max(xs_c))
            ax.plot(xr, [slope * x + intercept for x in xr],
                    color=C_TREND, linestyle=":", linewidth=0.9,
                    alpha=0.6, zorder=0, label="trend (cluster)")

    # 4) Annotate the two narrative-critical points (headline outlier + counter).
    for r in rows:
        name = display_model_name(r.profile)
        if not (_is_outlier(name) or _is_counter(name)):
            continue
        # Place labels to the side that keeps them out of the cluster.
        is_out = _is_outlier(name)
        colour = C_HIGHLIGHT if is_out else C_COUNTER
        # Outlier: bottom-right of plot → label up-left; counter (top): label up-right.
        if is_out:
            offset = (-30, -22) if r.mode == "agent" else (-30, 18)
        else:
            offset = (16, -10) if r.mode == "agent" else (16, 10)
        ax.annotate(
            f"{name} ({'agent' if r.mode == 'agent' else 'baseline'})",
            xy=(r.mean_recall, r.arena_rating),
            xytext=offset,
            textcoords="offset points",
            fontsize=7.5, color=colour, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=colour,
                            linewidth=0.7, alpha=0.6,
                            connectionstyle="arc3,rad=0.15"),
        )

    # 5) Spearman ρ in upper-left corner — uses ALL points (outlier included).
    xs_all = [r.mean_recall for r in rows]
    ys_all = [r.arena_rating for r in rows]
    rho = spearman_rho(xs_all, ys_all)
    ax.text(0.03, 0.97,
            f"Spearman $\\rho$ = {rho:.2f}  ($n$ = {len(rows)})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=8, color="#444444",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#DDDDDD", linewidth=0.6, alpha=0.92))

    # Axes.
    ax.set_xlabel("Reference recall")
    ax.set_ylabel("Arena BTD rating")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.5)

    # Legend.
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_AGENT,
               markersize=7, label="Agent", markeredgecolor="white",
               markeredgewidth=0.5),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=C_BASELINE,
               markersize=6.5, label="Baseline", markeredgecolor="white",
               markeredgewidth=0.5),
        Line2D([0], [0], color=C_PAIR_LINE, linewidth=0.8,
               label="Same Model Pair"),
    ]
    ax.legend(handles=legend_elements, loc="upper left",
              bbox_to_anchor=(0.03, 0.88),
              fontsize=8, frameon=False, handletextpad=0.5)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"wrote {out_path}")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out-dir", default=str(PAPER_IMAGES_DIR))
    parser.add_argument("--out-template", default="arena-vs-recall-{judge}.pdf")
    parser.add_argument("--min-cases", type=int, default=50,
                        help="Drop (model, mode) entries with fewer recall-evaluated cases.")
    parser.add_argument(
        "--arena-domains", choices=tuple(_DOMAIN_SETS), default="realworld",
        help="Domain set feeding the arena rating axis (default 'realworld' for "
             "apples-to-apples scope; mirrors paper_recall default).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    arena_domains = _DOMAIN_SETS[args.arena_domains]
    out_dir = Path(args.out_dir)
    for judge in args.judges:
        rows = _build_rows(
            judge, args.config,
            min_cases=args.min_cases, arena_domains=arena_domains,
        )
        if not rows:
            print(f"[{judge}] no eligible rows; skipping", file=sys.stderr)
            continue

        target = out_dir / args.out_template.format(judge=judge_short_name(judge))
        _render(rows, out_path=target, judge=judge,
                arena_domain_set=args.arena_domains)

        print()
        print(f"[{judge}] plotted {len(rows)} (model, mode) points; "
              f"Spearman ρ = "
              f"{spearman_rho([r.mean_recall for r in rows], [r.arena_rating for r in rows]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
