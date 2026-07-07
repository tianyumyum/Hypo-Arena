"""Standalone progress monitor: derives per-stage per-domain progress from artifacts/ files.

Run in a separate terminal alongside orchestrate.bash. Pure file-based: no IPC with the
orchestrator, just JSONL line counts (mtime-cached). Refreshes every --refresh seconds.

Stage totals are theoretical ceilings derived from CLI flags (must match orchestrate.bash):
- construction: total = N source records per domain
- generation:   total = N sources × n_modes × n_profiles per domain
- arena:        total = N sources × C(1 + n_modes × n_profiles, 2) per domain
- score:        total = N sources × (1 + n_modes × n_profiles) per domain
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from rich.columns import Columns
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from basics import ALL_DOMAINS
from basics.io import source_metadata_path
from basics.paths import (
    arena_matches_path,
    cases_path,
    score_records_path,
    submission_path,
)


DEFAULT_GEN_PROFILES = (
    "claude-sonnet-4.6-high", "claude-opus-4.6-high",
    "deepseek-v4-flash-high", "deepseek-v4-pro-high",
    "gemini-3-flash-high", "gemini-3.1-pro-high",
    "glm-5-thinking", "glm-5.1-thinking",
    "gpt-5.4-mini-high", "gpt-5.4-high",
    "kimi-k2.5-thinking", "kimi-k2.6-thinking",
    "minimax-m2.5-thinking", "minimax-m2.7-thinking",
    "qwen-3.6-max-thinking",
    "qwen-3.7-max-thinking",
)

_count_cache: dict[Path, tuple[float, int]] = {}


def count_lines(path: Path) -> int:
    """Non-empty line count in a JSONL; mtime-cached so re-reads are free when unchanged."""
    if not path.exists():
        return 0
    mtime = path.stat().st_mtime
    cached = _count_cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    n = 0
    with path.open("rb") as f:
        for line in f:
            if line.strip():
                n += 1
    _count_cache[path] = (mtime, n)
    return n


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _parse_domains(value: str) -> tuple[str, ...]:
    if value.strip().lower() == "all":
        return tuple(ALL_DOMAINS)
    out = _csv(value)
    bad = [d for d in out if d not in ALL_DOMAINS]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown domains: {bad}")
    return out


def _bar(done: int, total: int, width: int = 18) -> Text:
    """Unicode progress bar; green filled / dim unfilled."""
    if total <= 0:
        return Text(" " * width, style="dim")
    pct = done / total
    filled = max(0, min(width, round(pct * width)))
    return Text("█" * filled, style="green") + Text("░" * (width - filled), style="dim")


def _pct_str(done: int, total: int) -> str:
    if total <= 0:
        return "   -  "
    return f"{done / total * 100:5.1f}%"


def _fmt_int(n: int) -> str:
    return f"{n:,}"


def _stage_table(
    *,
    title: str,
    domains: tuple[str, ...],
    done_per_domain: dict[str, int],
    total_per_domain: dict[str, int],
    bar_width: int = 10,
) -> Table:
    """Per-domain progress table for one stage (or one stage×mode slice)."""
    # Explicit min_width on every column so columns line up across all 5 stage tables
    # (otherwise rich auto-sizes per table by content, e.g. arena's 538,032 makes
    # its `total` column wider than construction's 1,019).
    table = Table(title=title, title_style="bold cyan", expand=False, pad_edge=False)
    table.add_column("Domain", style="white", no_wrap=True, min_width=20)
    table.add_column("done", justify="right", style="green", no_wrap=True, min_width=7)
    table.add_column("total", justify="right", no_wrap=True, min_width=7)
    table.add_column("%", justify="right", no_wrap=True, min_width=6)
    table.add_column("progress", no_wrap=True, min_width=bar_width)

    sum_done = 0
    sum_total = 0
    for d in domains:
        done = done_per_domain[d]
        total = total_per_domain[d]
        sum_done += done
        sum_total += total
        table.add_row(
            d, _fmt_int(done), _fmt_int(total), _pct_str(done, total),
            _bar(done, total, width=bar_width),
        )
    table.add_row(
        Text("TOTAL", style="bold"),
        Text(_fmt_int(sum_done), style="bold green"),
        Text(_fmt_int(sum_total), style="bold"),
        Text(_pct_str(sum_done, sum_total), style="bold"),
        _bar(sum_done, sum_total, width=bar_width),
    )
    return table


def render(args, started: float) -> Group:
    elapsed = time.time() - started
    domains = args.domains
    profiles = args.gen_profiles
    modes = args.gen_modes
    judges = args.judges
    n_triples = len(modes) * len(profiles)
    n_labels = 1 + n_triples                                        # reference + each (mode, profile)
    pairs_per_case = n_labels * (n_labels - 1) // 2

    src_totals = {d: count_lines(source_metadata_path(d)) for d in domains}
    construction_done = {d: count_lines(cases_path(d, args.construction_profile)) for d in domains}

    header = Text.assemble(
        ("HypoArena Monitor", "bold"),
        ("    elapsed ", "dim"), (_fmt_duration(elapsed), "white"),
        ("    config=", "dim"), (args.construction_profile, "cyan"),
        ("    judges=", "dim"), (",".join(judges), "cyan"),
        ("    triples=", "dim"), (f"{n_triples}", "cyan"),
        ("    refresh ", "dim"), (f"{args.refresh:.0f}s", "white"),
    )

    # Generation: one sub-table per mode (baseline / agent), side-by-side.
    gen_tables: list[Table] = []
    for mode in modes:
        gen_done = {
            d: sum(
                count_lines(submission_path(d, mode, args.construction_profile, p))
                for p in profiles
            )
            for d in domains
        }
        gen_total = {d: src_totals[d] * len(profiles) for d in domains}
        gen_tables.append(_stage_table(
            title=f"Generation · {mode} (cases → submissions, ×{len(profiles)} profiles)",
            domains=domains,
            done_per_domain=gen_done, total_per_domain=gen_total,
        ))

    # Evaluation: per judge, one arena + one score table side-by-side.
    eval_rows: list[Columns] = []
    for judge in judges:
        arena_done = {
            d: count_lines(arena_matches_path(d, args.construction_profile, judge))
            for d in domains
        }
        arena_total = {d: src_totals[d] * pairs_per_case for d in domains}
        score_done = {
            d: count_lines(score_records_path(d, args.construction_profile, judge))
            for d in domains
        }
        score_total = {d: src_totals[d] * n_labels for d in domains}
        eval_rows.append(Columns(
            [
                _stage_table(
                    title=f"Evaluation · arena · {judge} (×{pairs_per_case} pairs per source)",
                    domains=domains,
                    done_per_domain=arena_done, total_per_domain=arena_total,
                ),
                _stage_table(
                    title=f"Evaluation · score · {judge} (×{n_labels} labels per source)",
                    domains=domains,
                    done_per_domain=score_done, total_per_domain=score_total,
                ),
            ],
            equal=True, expand=False, padding=(0, 2),
        ))

    return Group(
        Panel(header, border_style="cyan", expand=True),
        _stage_table(
            title="Construction (sources → cases)",
            domains=domains,
            done_per_domain=construction_done, total_per_domain=src_totals,
        ),
        Columns(gen_tables, equal=True, expand=False, padding=(0, 2)),
        *eval_rows,
    )


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="HypoArena standalone progress monitor (file-based, no orchestrator IPC)",
    )
    parser.add_argument("--domains", type=_parse_domains, default="all",
                        help="comma-list or 'all' (must match orchestrator)")
    parser.add_argument("--construction-profile", default="gpt-5.4")
    parser.add_argument("--gen-profiles", type=_csv, default=",".join(DEFAULT_GEN_PROFILES),
                        help="must match orchestrator's --gen-profiles")
    parser.add_argument("--gen-modes", type=_csv, default="baseline,agent")
    parser.add_argument("--judges", type=_csv, default="mimo-v2-pro",
                        help="comma-list of judge profiles (must match orchestrator's --judges)")
    parser.add_argument("--refresh", type=float, default=5.0,
                        help="refresh interval in seconds (default 5)")
    parser.add_argument("--once", action="store_true",
                        help="print one snapshot and exit (for non-TTY / cron usage)")

    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    started = time.time()

    if args.once or not sys.stdout.isatty():
        Console().print(render(args, started))
        return 0

    refresh_hz = max(0.1, 1.0 / max(args.refresh, 0.5))
    with Live(render(args, started), refresh_per_second=refresh_hz, screen=False) as live:
        try:
            while True:
                time.sleep(args.refresh)
                live.update(render(args, started))
        except KeyboardInterrupt:
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
