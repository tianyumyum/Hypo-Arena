"""Render Figure 2 of §4.2: Agent vs. Baseline arena ratings per model.

For every model evaluated under both modes, compute the cross-domain mean BTD
rating in baseline mode and in agent mode, then draw paired bars sorted by
baseline rating (descending). Each bar pair is annotated with its absolute ELO
delta (Δ); positive deltas are tinted green, negative red.

The reference horizontal line marks the Reference entry's cross-domain mean,
making it easy to see which models clear the static reference under either
mode.

Usage:
  uv run python -m scripts.paper_baseline_vs_agent [--judge mimo-v2-pro]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    ORDERED_DOMAINS,
    PAPER_AGENT_FIG_EXCLUDE,
    PAPER_IMAGES_DIR,
    PAPER_JUDGES,
    PRIMARY_JUDGE,
    REFERENCE_LABEL,
    LeaderboardRow,
    display_model_name,
    judge_short_name,
    load_arena_leaderboard,
    rows_by_profile,
)


def _mean_rating(by_domain: dict[str, dict[str, LeaderboardRow]],
                 profile: str) -> float | None:
    vals: list[float] = []
    for d in ORDERED_DOMAINS:
        row = by_domain[d].get(profile)
        if row is not None:
            vals.append(row.rating)
    return (sum(vals) / len(vals)) if vals else None


def _collect(judge: str, config: str, *, exclude: tuple[str, ...] = ()):
    """Read full-pool leaderboards so baseline:X and agent:X live in one BTD scale."""
    domain_rows = {
        d: load_arena_leaderboard(d, judge=judge, config=config, pool="full")
        for d in ORDERED_DOMAINS
    }
    baseline_by_d = {d: rows_by_profile(domain_rows[d], mode="baseline") for d in ORDERED_DOMAINS}
    agent_by_d = {d: rows_by_profile(domain_rows[d], mode="agent") for d in ORDERED_DOMAINS}

    profiles = sorted(set().union(*(set(b) for b in baseline_by_d.values())))
    excluded = set(exclude)
    rows = []
    for profile in profiles:
        if profile in excluded:
            continue
        b = _mean_rating(baseline_by_d, profile)
        a = _mean_rating(agent_by_d, profile)
        if b is None or a is None:
            continue                                    # only keep models present in both modes
        rows.append((profile, b, a))
    rows.sort(key=lambda x: -x[1])                      # by baseline desc

    # Reference cross-domain mean (also from the full pool, so it shares scale).
    ref_vals: list[float] = []
    for d in ORDERED_DOMAINS:
        for r in domain_rows[d]:
            if r.raw_label == REFERENCE_LABEL:
                ref_vals.append(r.rating)
                break
    reference_mean = sum(ref_vals) / len(ref_vals) if ref_vals else None
    return rows, reference_mean


def _render(rows, reference_mean, *, out_path: Path, judge: str) -> None:
    labels = [display_model_name(p) for p, _, _ in rows]
    baseline_vals = np.array([b for _, b, _ in rows])
    agent_vals = np.array([a for _, _, a in rows])

    n = len(rows)
    x = np.arange(n)
    bar_w = 0.38

    fig_w = max(8.0, 0.9 * n + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, 4.6), dpi=200)

    # Single-hue blue palette (Agent lighter, Baseline darker).
    color_agent = "#cfdbf0"
    color_baseline = "#a0b7e1"

    # v2 visual ordering: Agent on the LEFT half of each pair, Baseline on the right.
    bars_a = ax.bar(x - bar_w / 2, agent_vals, width=bar_w,
                    color=color_agent, label="Agent", edgecolor="white")
    bars_b = ax.bar(x + bar_w / 2, baseline_vals, width=bar_w,
                    color=color_baseline, label="Baseline", edgecolor="white")

    # Reference line.
    ref_line = None
    if reference_mean is not None:
        ref_line = ax.axhline(reference_mean, color="#888888", linestyle="--",
                              linewidth=1.0, label="Reference")

    # Δ annotations above each pair: green for positive, red for negative.
    pair_max = np.maximum(baseline_vals, agent_vals)
    for i, (b, a, top) in enumerate(zip(baseline_vals, agent_vals, pair_max)):
        delta = a - b
        sign = "+" if delta >= 0 else "-"
        color = "#2E7D32" if delta >= 0 else "#C62828"
        ax.text(i, top + 6, f"{sign}{abs(delta):.0f}", ha="center",
                va="bottom", fontsize=8, color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Average Arena ELO Rating", fontsize=9)

    y_min = float(min(baseline_vals.min(), agent_vals.min()) - 30)
    y_max = float(pair_max.max() + 60)
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # Spines off (cleaner v2-style), legend top-right without frame.
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend_handles = [bars_a, bars_b] + ([ref_line] if ref_line is not None else [])
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out-dir", default=str(PAPER_IMAGES_DIR))
    parser.add_argument("--out-template", default="baseline-vs-agent-{judge}.pdf")
    parser.add_argument(
        "--exclude", default=",".join(PAPER_AGENT_FIG_EXCLUDE),
        help=("Comma-list of profiles to drop from the figure. Default excludes the qwen "
              "family for paper readability; pass empty string to keep all."),
    )
    parser.add_argument(
        "--include-all", action="store_true",
        help="Shortcut for --exclude '' — render every model present in both modes.",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    exclude_raw = "" if args.include_all else args.exclude
    exclude = tuple(s.strip() for s in exclude_raw.split(",") if s.strip())

    out_dir = Path(args.out_dir)
    for judge in args.judges:
        rows, reference_mean = _collect(judge, args.config, exclude=exclude)
        target = out_dir / args.out_template.format(judge=judge_short_name(judge))
        _render(rows, reference_mean, out_path=target, judge=judge)

        print()
        print(f"[{judge}] Per-model deltas (Agent − Baseline, cross-domain mean):")
        for p, b, a in rows:
            delta = a - b
            sign = "+" if delta >= 0 else "−"
            print(f"  {display_model_name(p):<28s}  baseline={b:7.1f}  agent={a:7.1f}  "
                  f"Δ {sign}{abs(delta):5.1f}")
        if reference_mean is not None:
            print(f"  Reference                    mean = {reference_mean:7.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
