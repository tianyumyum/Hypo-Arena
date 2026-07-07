"""Render a cross-domain baseline-pool arena leaderboard as Markdown.

Companion to ``paper_arena_all.py`` (per-judge LaTeX). Per-domain BTD ratings
plus an Avg-BTD sort key and a cross-domain Win Rate. One file per run, both
judges in the same file.

Output: ``tables/arena-cross-baseline.md`` (under hypocode-v2/)

Columns:
  Rank | Model | <Domain1 BTD> | ... | <Domain6 BTD> | Avg BTD | Win Rate

  - Each domain column is that model's BTD rating computed inside that domain's
    baseline pool (independent BTD per domain — comparable to per-domain
    leaderboards already published).
  - Avg BTD = simple mean over the (up to) 6 per-domain BTDs the model has.
  - Win Rate = aggregate across all 6 domains' baseline matches:
        (wins + 0.5 * ties) / (wins + losses + ties)
    where the result is decided by the debiased score (forward + 1-reverse) / 2.
  - Rows sorted by Avg BTD descending.

Usage:
  uv run python -m scripts.paper_arena_cross_baseline
  uv run python -m scripts.paper_arena_cross_baseline --judges mimo-v2-pro
  uv run python -m scripts.paper_arena_cross_baseline --out /tmp/x.md
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import REFERENCE_LABEL
from basics.io import iter_jsonl
from basics.paths import ARTIFACTS_ROOT
from basics.schema import ArenaMatch, keep_in_pool
from evaluation.arena import compute_btd
from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    DOMAIN_SHORT_LABEL,
    ORDERED_DOMAINS,
    PAPER_JUDGES,
    display_model_name,
    judge_short_name,
)

POOL = "baseline"


def _load_matches(domain: str, judge: str, config: str) -> list[ArenaMatch]:
    path = (
        ARTIFACTS_ROOT / domain / "results"
        / f"{config}.{judge}.arena.matches.jsonl"
    )
    return list(iter_jsonl(path, ArenaMatch))


def _filter_pool(matches: list[ArenaMatch]) -> list[ArenaMatch]:
    return [m for m in matches
            if keep_in_pool(m.model_a, POOL) and keep_in_pool(m.model_b, POOL)]


def _aggregate_winrate(matches: list[ArenaMatch]) -> dict[str, float]:
    """Per-model winrate over the supplied match list. Win counts use debiased score:
    score > 0.5 → A wins, < 0.5 → B wins, == 0.5 → tie.
    winrate = (wins + 0.5*ties) / (wins + losses + ties).
    """
    stats: dict[str, dict[str, int]] = defaultdict(lambda: {"w": 0, "l": 0, "t": 0})
    for m in matches:
        s = m.debiased_score
        a, b = m.model_a, m.model_b
        if s > 0.5:
            stats[a]["w"] += 1
            stats[b]["l"] += 1
        elif s < 0.5:
            stats[b]["w"] += 1
            stats[a]["l"] += 1
        else:
            stats[a]["t"] += 1
            stats[b]["t"] += 1
    out: dict[str, float] = {}
    for model, c in stats.items():
        total = c["w"] + c["l"] + c["t"]
        if total:
            out[model] = (c["w"] + 0.5 * c["t"]) / total
    return out


def _row_label(raw: str) -> str:
    if raw == REFERENCE_LABEL:
        return "**reference**"
    profile = raw.split(":", 1)[1] if ":" in raw else raw
    return f"`{display_model_name(profile)}`"


def _fmt_btd(value: float | None) -> str:
    return "—" if value is None else f"{value:.1f}"


def _fmt_winrate(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.1f}%"


def _render_judge(judge: str, config: str) -> str:
    """Build the Markdown section for one judge."""
    # Per-domain BTD + per-model presence
    per_domain_ratings: dict[str, dict[str, float]] = {}
    per_domain_pool_matches: dict[str, list[ArenaMatch]] = {}
    for d in ORDERED_DOMAINS:
        pool_ms = _filter_pool(_load_matches(d, judge, config))
        per_domain_pool_matches[d] = pool_ms
        per_domain_ratings[d] = compute_btd(pool_ms) if pool_ms else {}

    # Union of models that appear in any domain
    all_models: set[str] = set()
    for ratings in per_domain_ratings.values():
        all_models.update(ratings)

    if not all_models:
        return f"## judge: `{judge}`\n\n_no matches in pool=`{POOL}`._\n"

    # Cross-domain aggregate winrate (one number per model)
    all_pool_matches: list[ArenaMatch] = []
    for ms in per_domain_pool_matches.values():
        all_pool_matches.extend(ms)
    cross_winrate = _aggregate_winrate(all_pool_matches)

    # Build rows: list of (display, raw_label, per_domain_btd_list, avg_btd, winrate)
    rows: list[tuple[str, str, list[float | None], float, float | None]] = []
    for model in all_models:
        per_dom = [per_domain_ratings[d].get(model) for d in ORDERED_DOMAINS]
        present = [v for v in per_dom if v is not None]
        if not present:
            continue
        avg_btd = sum(present) / len(present)
        wr = cross_winrate.get(model)
        rows.append((_row_label(model), model, per_dom, avg_btd, wr))

    # Sort by avg BTD descending
    rows.sort(key=lambda r: -r[3])

    short_judge = judge_short_name(judge)
    short_dom = [DOMAIN_SHORT_LABEL[d] for d in ORDERED_DOMAINS]
    pool_match_count = len(all_pool_matches)

    lines: list[str] = []
    lines.append(f"## judge: `{judge}` ({short_judge})")
    lines.append("")
    lines.append(
        f"- **Pool**: `{POOL}` &nbsp;·&nbsp; "
        f"**Models**: {len(rows)} &nbsp;·&nbsp; "
        f"**Matches**: {pool_match_count:,}"
    )
    lines.append("")

    head = ["Rank", "Model"] + short_dom + ["Avg BTD", "Win Rate"]
    align = ["---:", ":---"] + ["---:"] * len(short_dom) + ["---:", "---:"]
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|" + "|".join(align) + "|")

    for rank, (display, _raw, per_dom, avg_btd, wr) in enumerate(rows, start=1):
        cells = [str(rank), display]
        cells.extend(_fmt_btd(v) for v in per_dom)
        cells.append(_fmt_btd(avg_btd))
        cells.append(_fmt_winrate(wr))
        lines.append("| " + " | ".join(cells) + " |")

    return "\n".join(lines) + "\n"


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--judges", type=_csv, default=",".join(PAPER_JUDGES))
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument("--out", default="tables/arena-cross-baseline.md",
                        help="output path, relative to hypocode-v2/")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    sections: list[str] = [
        "# Cross-Domain Baseline Arena Leaderboard",
        "",
        f"Construction profile: `{args.config}` &nbsp;·&nbsp; Pool: `{POOL}` "
        "&nbsp;·&nbsp; Domains: 6 (research + real-world).",
        "",
        "Each domain column is the model's BTD rating computed within that "
        "domain's baseline pool. *Avg BTD* is the mean across the model's "
        "available domain BTDs (sort key, descending). *Win Rate* aggregates "
        "all 6 domains' baseline matches: (wins + 0.5·ties) / total.",
        "",
    ]
    for judge in args.judges:
        sections.append(_render_judge(judge, args.config))
        sections.append("---")
        sections.append("")

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = _HYPO_ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(sections), encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
