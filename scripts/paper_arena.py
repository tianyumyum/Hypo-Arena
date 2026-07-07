"""Render the §4.2 Main Results leaderboard (`tables/arena-{judge}.tex`).

For every baseline profile evaluated under the primary judge, the script
collects per-domain BTD ratings + win rates, averages them across present
domains, and emits the LaTeX table the paper expects (six per-domain columns
plus Avg + WR, with the Reference row inserted at its true rank).

Usage:
  uv run python -m scripts.paper_arena [--judge mimo-v2-pro]
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from scripts._paper_common import (
    CONSTRUCTION_PROFILE,
    DOMAIN_SHORT_LABEL,
    ORDERED_DOMAINS,
    PAPER_JUDGES,
    PAPER_MAIN_TABLE_MODELS,
    PAPER_TABLES_DIR,
    REFERENCE_LABEL,
    LeaderboardRow,
    display_model_name,
    judge_short_name,
    load_arena_leaderboard,
    load_restricted_arena_leaderboard,
    patch_table_body,
    rows_by_profile,
    write_text_artifact,
)


@dataclass
class ModelSummary:
    """One row of the main results table (a baseline profile or Reference)."""

    display_name: str
    is_reference: bool
    per_domain_rating: dict[str, float]   # domain → BTD rating (only present ones)
    per_domain_winrate: dict[str, float]  # domain → WR ∈ [0, 1]
    profile: str

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


def _collect(
    judge: str,
    config: str,
    *,
    whitelist: tuple[str, ...] | None,
    restricted_btd: bool,
) -> tuple[list[ModelSummary], ModelSummary]:
    """Load all six domains' baseline-pool leaderboards, build per-profile summaries.

    When ``restricted_btd=True`` and ``whitelist`` is set, recompute BTD from raw
    matches restricted to exactly {whitelist baselines} ∪ {reference}. This matches
    a strict reading of paper §4.2 ("Table 1 is Baseline Mode over the 14 evaluated
    models") where the BTD competition set equals the display set.

    When ``restricted_btd=False`` (default), ratings come from the pool=baseline
    leaderboard which includes every baseline profile present in data — consistent
    with §4.3 / §4.4 stats that read the same pool file.
    """
    if restricted_btd and whitelist is not None:
        allow_labels = {f"baseline:{p}" for p in whitelist} | {REFERENCE_LABEL}
        by_domain: dict[str, list[LeaderboardRow]] = {
            d: load_restricted_arena_leaderboard(
                d, allow_labels=allow_labels, judge=judge, config=config,
            )
            for d in ORDERED_DOMAINS
        }
    else:
        by_domain = {
            d: load_arena_leaderboard(d, judge=judge, config=config, pool="baseline")
            for d in ORDERED_DOMAINS
        }

    discovered: set[str] = set()
    for rows in by_domain.values():
        discovered |= {r.profile for r in rows if r.mode == "baseline"}

    if whitelist is not None:
        baseline_profiles = [p for p in whitelist if p in discovered]
        missing = [p for p in whitelist if p not in discovered]
        if missing:
            print(f"warning: whitelist profiles missing from leaderboards: {missing}")
    else:
        baseline_profiles = sorted(discovered)

    summaries: list[ModelSummary] = []
    for profile in baseline_profiles:
        per_rating: dict[str, float] = {}
        per_wr: dict[str, float] = {}
        for d in ORDERED_DOMAINS:
            row = rows_by_profile(by_domain[d], mode="baseline").get(profile)
            if row is None:
                continue
            per_rating[d] = row.rating
            wr = row.win_rate
            if wr is not None:
                per_wr[d] = wr
        summaries.append(
            ModelSummary(
                display_name=display_model_name(profile),
                is_reference=False,
                per_domain_rating=per_rating,
                per_domain_winrate=per_wr,
                profile=profile,
            )
        )

    # Reference appears in every domain; pull its row from each.
    ref_rating: dict[str, float] = {}
    ref_wr: dict[str, float] = {}
    for d in ORDERED_DOMAINS:
        for row in by_domain[d]:
            if row.raw_label == REFERENCE_LABEL:
                ref_rating[d] = row.rating
                if row.win_rate is not None:
                    ref_wr[d] = row.win_rate
                break
    reference = ModelSummary(
        display_name="reference",
        is_reference=True,
        per_domain_rating=ref_rating,
        per_domain_winrate=ref_wr,
        profile=REFERENCE_LABEL,
    )

    summaries.sort(key=lambda s: (s.avg_rating is None, -(s.avg_rating or 0.0)))
    return summaries, reference


# ---- LaTeX rendering -----------------------------------------------------

def _fmt_rating(value: float | None, *, bold: bool = False) -> str:
    if value is None:
        return "—"
    text = f"{value:.1f}"
    return f"\\textbf{{{text}}}" if bold else text


def _fmt_winrate(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}\\%"


def _thinking_effort(profile_name: str) -> str:
    """Map profile suffix to thinking-effort label.

    `*-{xhigh,high,medium,low,minimal}` returns the suffix verbatim;
    `*-thinking` returns ``"enabled"``; anything else returns ``"—"``.
    """
    for suffix in ("xhigh", "high", "medium", "low", "minimal"):
        if profile_name.endswith(f"-{suffix}"):
            return suffix
    if profile_name.endswith("-thinking"):
        return r"\checkmark"
    return "—"


def _row_for(
    summary: ModelSummary,
    *,
    rank_label: str,
    column_max_per_domain: dict[str, float],
    column_max_avg: float,
    italic: bool = False,
    rowcolor: str | None = None,
    effort_cell: str | None = None,
) -> str:
    cells: list[str] = []
    if rowcolor is not None:
        cells.append(f"\\rowcolor{{{rowcolor}}} {rank_label}")
    else:
        cells.append(rank_label)

    name_cell = f"\\textit{{{summary.display_name}}}" if italic else summary.display_name
    cells.append(name_cell)

    if effort_cell is None:
        effort_cell = _thinking_effort(summary.profile)
    cells.append(f"\\textit{{{effort_cell}}}" if italic else effort_cell)

    for d in ORDERED_DOMAINS:
        rating = summary.per_domain_rating.get(d)
        bold = rating is not None and rating == column_max_per_domain.get(d)
        cell = _fmt_rating(rating, bold=bold)
        cells.append(f"\\textit{{{cell}}}" if italic and not bold else cell)

    avg = summary.avg_rating
    bold_avg = avg is not None and avg == column_max_avg
    cells.append(f"\\textit{{{_fmt_rating(avg, bold=bold_avg)}}}" if italic and not bold_avg
                 else _fmt_rating(avg, bold=bold_avg))
    cells.append(f"\\textit{{{_fmt_winrate(summary.avg_winrate)}}}" if italic
                 else _fmt_winrate(summary.avg_winrate))

    return " & ".join(cells) + " \\\\"


ReferencePosition = Literal["top", "bottom", "rank"]


def _build_body(
    summaries: list[ModelSummary],
    reference: ModelSummary,
    *,
    reference_position: ReferencePosition = "bottom",
) -> str:
    """Build only the body rows (between first \\midrule and \\bottomrule).

    Used by `render_table` (full output) and `--update-rows-only` (in-place body
    patch that preserves caption / label / column headers).
    """
    # Bold the per-domain max across baselines + reference combined.
    all_models = summaries + [reference]
    column_max_per_domain: dict[str, float] = {}
    for d in ORDERED_DOMAINS:
        vals = [m.per_domain_rating[d] for m in all_models if d in m.per_domain_rating]
        if vals:
            column_max_per_domain[d] = max(vals)
    avg_vals = [m.avg_rating for m in all_models if m.avg_rating is not None]
    column_max_avg = max(avg_vals) if avg_vals else 0.0

    def _ref_row(rank_label: str = "—") -> str:
        return _row_for(
            reference,
            rank_label=rank_label,
            column_max_per_domain=column_max_per_domain,
            column_max_avg=column_max_avg,
            italic=False,
            rowcolor="colorGT",
            effort_cell="",
        )

    body_lines: list[str] = []

    if reference_position == "top":
        body_lines.append(_ref_row())
        body_lines.append("\\midrule")

    if reference_position == "rank":
        combined = sorted(
            [*summaries, reference],
            key=lambda s: -(s.avg_rating if s.avg_rating is not None else float("-inf")),
        )
        for i, summary in enumerate(combined):
            if summary is reference:
                body_lines.append(_ref_row(rank_label=str(i + 1)))
            else:
                body_lines.append(_row_for(
                    summary,
                    rank_label=str(i + 1),
                    column_max_per_domain=column_max_per_domain,
                    column_max_avg=column_max_avg,
                ))
    else:
        # top / bottom: emit baselines as a contiguous block.
        for i, summary in enumerate(summaries):
            body_lines.append(_row_for(
                summary,
                rank_label=str(i + 1),
                column_max_per_domain=column_max_per_domain,
                column_max_avg=column_max_avg,
            ))

    if reference_position == "bottom":
        body_lines.append("\\midrule")
        body_lines.append(_ref_row())

    return "\n".join(body_lines)


def render_table(
    summaries: list[ModelSummary],
    reference: ModelSummary,
    *,
    judge: str,
    reference_position: ReferencePosition = "bottom",
) -> str:
    """Render Table 1 in LaTeX (full table including caption + label).

    ``reference_position`` controls where the Reference row appears:
      * ``top``    — first row, before all baselines
      * ``bottom`` — last row, after all baselines (default; reads as a footer anchor)
      * ``rank``   — slot in by Avg rating, with surrounding midrules
    """
    body = _build_body(summaries, reference, reference_position=reference_position)
    headers = " & ".join(["\\textbf{" + DOMAIN_SHORT_LABEL[d] + "}" for d in ORDERED_DOMAINS])
    n_baselines = len(summaries)
    caption = (
        f"Main leaderboard under the reference-free arena setting (Baseline Mode), "
        f"judged by \\texttt{{{judge}}}. BTD Arena ratings (centered at 1500) are reported "
        f"across six evaluated domains---three research domains (Biomedical Science, "
        f"Machine Learning, Social Science) and three real-world domains (Financial Analysis, "
        f"IT Operations, Safety Investigation). Higher scores indicate superior hypothesis "
        f"generation. WinRate denotes the average fraction of pairwise matchups won. "
        f"\\colorbox{{colorGT}}{{Reference}} serves as an anonymous competitor for calibration. "
        f"({n_baselines} baseline models.)"
    )

    return (
        "\\begin{table*}[t]\n"
        "\\centering\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{tab:arena-{judge_short_name(judge)}}}\n"
        "\\small\n"
        "\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\begin{tabular}{cll rrr rrr rr}\n"
        "\\toprule\n"
        " & & & \\multicolumn{3}{c}{\\textbf{Research}} & \\multicolumn{3}{c}{\\textbf{Real-World}} & & \\\\\n"
        "\\cmidrule(lr){4-6} \\cmidrule(lr){7-9}\n"
        f"\\textbf{{\\#}} & \\textbf{{Model}} & \\textbf{{Effort}} & {headers} & \\textbf{{Avg}} & \\textbf{{WR}} \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table*}\n"
    )


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _render_one_judge(
    judge: str,
    config: str,
    *,
    whitelist: tuple[str, ...] | None,
    restricted_btd: bool,
    reference_position: ReferencePosition,
    target: Path,
    update_rows_only: bool = True,
) -> None:
    """Build, render, write, and print one judge's Table 1.

    If ``update_rows_only`` is True (default), only the body rows of an existing
    target file are replaced; caption and label are preserved. Falls back to a
    full write when the target file does not yet exist.
    """
    summaries, reference = _collect(
        judge, config, whitelist=whitelist, restricted_btd=restricted_btd,
    )
    if update_rows_only and target.exists():
        body = _build_body(
            summaries, reference, reference_position=reference_position,
        )
        patch_table_body(target, body)
    else:
        text = render_table(
            summaries, reference, judge=judge, reference_position=reference_position,
        )
        write_text_artifact(target, text)

    print()
    print(f"Ranked baselines (judge={judge}):")
    for i, s in enumerate(summaries, 1):
        avg = "—" if s.avg_rating is None else f"{s.avg_rating:7.1f}"
        wr = "—" if s.avg_winrate is None else f"{s.avg_winrate * 100:5.1f}%"
        present = len(s.per_domain_rating)
        print(f"  {i:>2}.  {s.display_name:<28s}  avg={avg}  WR={wr}  domains={present}/6")
    if reference.avg_rating is not None:
        print(f"  ref. {reference.display_name:<28s}  avg={reference.avg_rating:7.1f}  "
              f"WR={(reference.avg_winrate or 0)*100:5.1f}%")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judges", type=_csv, default=",".join(PAPER_JUDGES),
        help=("Comma-list of judge profiles. Default produces one Table 1 per judge, "
              "named arena-<short>.tex (mimo / seed)."),
    )
    parser.add_argument("--config", default=CONSTRUCTION_PROFILE)
    parser.add_argument(
        "--out-dir", default=str(PAPER_TABLES_DIR),
        help="Output directory for per-judge files (ignored if --out is given).",
    )
    parser.add_argument(
        "--out-template", default="arena-{judge}.tex",
        help="Filename template; {judge} is replaced by judge_short_name(judge).",
    )
    parser.add_argument(
        "--out", default=None,
        help=("Explicit single output path. Requires exactly one --judges. Overrides "
              "--out-dir / --out-template; useful for ad-hoc / test runs."),
    )
    parser.add_argument(
        "--no-whitelist", action="store_true",
        help="Render every baseline profile present in the leaderboards (skip "
             "PAPER_MAIN_TABLE_MODELS filter).",
    )
    parser.add_argument(
        "--restricted-btd", action="store_true",
        help=("Recompute BTD from raw arena.matches.jsonl restricted to whitelist + "
              "Reference (paper-strict reading: competition set = display set). "
              "Without this flag, ratings come from the pool=baseline leaderboard "
              "(consistent with §4.3 / §4.4 stats). Requires whitelist to be active."),
    )
    parser.add_argument(
        "--reference-position", choices=("top", "bottom", "rank"), default="rank",
        help=("Where to put the Reference row. 'rank' (default) ranks Reference "
              "alongside the baselines by Avg, with no separator midrules; "
              "'top'/'bottom' pin it as a calibration anchor at the respective end "
              "(separated by midrules)."),
    )
    parser.add_argument(
        "--update-rows-only", action=argparse.BooleanOptionalAction, default=True,
        help=("Patch only the body rows of an existing target file, preserving the "
              "caption / label / column-headers (default). Falls back to a full "
              "write when the target file does not exist. Pass --no-update-rows-only "
              "to overwrite the entire file (including caption); useful when the "
              "column structure changes."),
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    if not args.judges:
        parser.error("--judges must list at least one judge profile.")
    if args.out is not None and len(args.judges) != 1:
        parser.error(
            f"--out requires exactly one --judges (got {len(args.judges)}: {args.judges}). "
            f"Use --out-dir / --out-template for multi-judge runs."
        )
    if args.restricted_btd and args.no_whitelist:
        parser.error("--restricted-btd is incompatible with --no-whitelist; the "
                     "restricted-BTD path needs an explicit competition set.")

    whitelist = None if args.no_whitelist else PAPER_MAIN_TABLE_MODELS
    out_dir = Path(args.out_dir)

    for judge in args.judges:
        if args.out is not None:
            target = Path(args.out)
        else:
            target = out_dir / args.out_template.format(judge=judge_short_name(judge))
        _render_one_judge(
            judge, args.config,
            whitelist=whitelist, restricted_btd=args.restricted_btd,
            reference_position=args.reference_position, target=target,
            update_rows_only=args.update_rows_only,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
