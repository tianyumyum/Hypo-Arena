"""Probe every (profile, platform) combo with a minimal prompt to surface dead backends.

Usage:
    uv run python scripts/platform_probe.py                          # default: ALL profiles, both registries
    uv run python scripts/platform_probe.py --registry generation    # only generation registry
    uv run python scripts/platform_probe.py --profiles gpt-5.4,kimi-k2.6  # filter to specific profiles
    uv run python scripts/platform_probe.py --only-failing           # collapse healthy combos in output

For each (profile, platform), sends a one-line chat-completions request and reports:
- OK (elapsed, reply)
- N/A (env vars missing or model_id placeholder)
- ERR (exception class + short message)

Many profiles share the same (platform, model_id) tuple (e.g., gpt-5.4 / gpt-5.4-low /
gpt-5.4-high all hit the same backend). This script dedups the actual API calls so the
~50+ profile registry only fires ~30 unique probes.

Use this to:
- Find dead platform/model combos before launching orchestrator
- Diagnose why FallbackModel can't find a working backend for a given profile
- Compare per-platform health (which platform is most reliable right now)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from collections import defaultdict
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.text import Text

from basics.models import CONSTRUCTION_REGISTRY, GENERATION_REGISTRY
from basics.platform import get_client, is_available


TEST_PROMPT = "Reply with exactly one word: PONG"
TIMEOUT_SEC = 30.0
MAX_TOKENS = 32
PROBE_CONCURRENCY = 16


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


async def probe_one(platform: str, model_id: str) -> tuple[str, str, float]:
    """Return (status_tag, detail, elapsed_sec). status_tag ∈ {'ok', 'na', 'err'}."""
    if not is_available(platform):
        return ("na", "env vars missing", 0.0)
    client = get_client(platform)
    t0 = time.time()
    try:
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": TEST_PROMPT}],
                max_tokens=MAX_TOKENS,
            ),
            timeout=TIMEOUT_SEC,
        )
        elapsed = time.time() - t0
        text = (resp.choices[0].message.content or "").strip()
        return ("ok", repr(text), elapsed)
    except asyncio.TimeoutError:
        return ("err", f"TimeoutError: >{TIMEOUT_SEC}s", time.time() - t0)
    except Exception as exc:
        return ("err", f"{type(exc).__name__}: {str(exc)[:200]}", time.time() - t0)


def _collect_targets(
    *,
    registries: tuple[str, ...],
    profile_filter: tuple[str, ...] | None,
) -> list[tuple[str, str, str, str | None]]:
    """Build a flat list of (registry, profile_name, platform, model_id) targets."""
    targets: list[tuple[str, str, str, str | None]] = []
    name_to_registry = {
        "construction": ("construction", CONSTRUCTION_REGISTRY),
        "generation":   ("generation",   GENERATION_REGISTRY),
    }
    for reg_name in registries:
        if reg_name not in name_to_registry:
            continue
        _, registry = name_to_registry[reg_name]
        for prof_name, profile in registry.items():
            if profile_filter and prof_name not in profile_filter:
                continue
            for platform, model_id in profile.platform_ids:
                targets.append((reg_name, prof_name, platform, model_id))
    return targets


async def main_async(
    *,
    registries: tuple[str, ...],
    profile_filter: tuple[str, ...] | None,
    only_failing: bool,
) -> int:
    console = Console()

    targets = _collect_targets(registries=registries, profile_filter=profile_filter)
    if not targets:
        console.print("[red]no targets matched[/red]")
        return 1

    if profile_filter:
        unknown = [
            p for p in profile_filter
            if p not in CONSTRUCTION_REGISTRY and p not in GENERATION_REGISTRY
        ]
        for p in unknown:
            console.print(f"[red]unknown profile: {p}[/red]")

    # Dedup by (platform, model_id) — many profiles share the same backend tuple.
    unique_combos: set[tuple[str, str]] = {
        (pl, mid) for _, _, pl, mid in targets if mid is not None
    }
    console.print(
        f"Probing [bold cyan]{len(targets)}[/bold cyan] (profile, platform) combos "
        f"→ [bold cyan]{len(unique_combos)}[/bold cyan] unique (platform, model_id) backends "
        f"with a {TIMEOUT_SEC:.0f}s per-call timeout…\n"
    )

    sem = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def wrapped(plat: str, mid: str):
        async with sem:
            tag, detail, elapsed = await probe_one(plat, mid)
            return ((plat, mid), tag, detail, elapsed)

    probe_results = await asyncio.gather(
        *(wrapped(pl, mid) for pl, mid in unique_combos)
    )
    by_combo: dict[tuple[str, str], tuple[str, str, float]] = {
        key: (tag, detail, elapsed) for key, tag, detail, elapsed in probe_results
    }

    # Build per-target results by lookup (preserves cheap dedup).
    rows: list[tuple[str, str, str, str, str, str, float]] = []
    # (registry, profile, platform, model_id, status_tag, detail, elapsed)
    for reg_name, prof_name, platform, model_id in targets:
        if model_id is None:
            rows.append((reg_name, prof_name, platform, "(none)", "na",
                         "model_id not configured", 0.0))
            continue
        tag, detail, elapsed = by_combo[(platform, model_id)]
        rows.append((reg_name, prof_name, platform, model_id, tag, detail, elapsed))

    # ---- detail table grouped by profile ----
    style_by_tag = {"ok": "green", "err": "red", "na": "yellow"}
    sym_by_tag = {"ok": "✓", "err": "✗", "na": "·"}

    detail_table = Table(title="Per-Combo Probe", title_style="bold cyan", expand=False)
    detail_table.add_column("Reg", no_wrap=True, style="dim")
    detail_table.add_column("Profile", no_wrap=True, style="white")
    detail_table.add_column("Platform", no_wrap=True)
    detail_table.add_column("Model ID", no_wrap=True, overflow="ellipsis", max_width=30)
    detail_table.add_column("Status", no_wrap=True)
    detail_table.add_column("Time", justify="right", no_wrap=True)
    detail_table.add_column("Detail", overflow="ellipsis", max_width=60, no_wrap=True)

    current_profile = None
    for reg_name, prof_name, platform, model_id, tag, detail, elapsed in rows:
        if only_failing and tag == "ok":
            continue
        if prof_name != current_profile:
            if current_profile is not None:
                detail_table.add_section()
            current_profile = prof_name
        elapsed_str = f"{elapsed:.1f}s" if elapsed > 0 else "-"
        detail_table.add_row(
            reg_name, prof_name, platform, model_id,
            Text(f"{sym_by_tag[tag]} {tag.upper()}", style=style_by_tag[tag]),
            elapsed_str,
            Text(detail, style=style_by_tag[tag]),
        )
    if detail_table.row_count > 0:
        console.print(detail_table)
    elif only_failing:
        console.print("[bold green]All probed combos healthy.[/bold green]\n")

    # ---- per-profile summary ----
    console.print()
    profile_summary = Table(title="Per-Profile Health", title_style="bold cyan", expand=False)
    profile_summary.add_column("Reg", no_wrap=True, style="dim")
    profile_summary.add_column("Profile", no_wrap=True)
    profile_summary.add_column("OK / Total", justify="right")
    profile_summary.add_column("Healthy platforms", overflow="fold", max_width=30, style="green")
    profile_summary.add_column("Dead platforms", overflow="fold", max_width=30, style="red")

    by_profile: dict[tuple[str, str], list[tuple[str, str, str, str, str, str, float]]] = defaultdict(list)
    for row in rows:
        by_profile[(row[0], row[1])].append(row)

    n_dead = 0
    n_partial = 0
    n_full = 0
    for (reg_name, prof_name), prof_rows in by_profile.items():
        oks = [r for r in prof_rows if r[4] == "ok"]
        bads = [r for r in prof_rows if r[4] != "ok"]
        healthy = ", ".join(r[2] for r in oks) or "—"
        dead = ", ".join(r[2] for r in bads) or "—"
        total = len(prof_rows)
        n_ok = len(oks)
        if n_ok == 0:
            count_style = "bold red"
            n_dead += 1
        elif n_ok < total:
            count_style = "yellow"
            n_partial += 1
        else:
            count_style = "green"
            n_full += 1
        profile_summary.add_row(
            reg_name, prof_name,
            Text(f"{n_ok} / {total}", style=count_style),
            healthy,
            dead,
        )
    console.print(profile_summary)

    # ---- per-platform summary ----
    console.print()
    platform_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"ok": 0, "err": 0, "na": 0})
    for row in rows:
        platform_stats[row[2]][row[4]] += 1

    plat_summary = Table(title="Per-Platform Health (across all probed combos)",
                         title_style="bold cyan", expand=False)
    plat_summary.add_column("Platform", no_wrap=True)
    plat_summary.add_column("OK", justify="right", style="green")
    plat_summary.add_column("ERR", justify="right", style="red")
    plat_summary.add_column("N/A", justify="right", style="yellow")
    plat_summary.add_column("Total", justify="right")
    plat_summary.add_column("OK rate", justify="right")
    for platform in sorted(platform_stats):
        stats = platform_stats[platform]
        total = stats["ok"] + stats["err"] + stats["na"]
        rate = stats["ok"] / total if total else 0.0
        rate_style = "green" if rate > 0.7 else "yellow" if rate > 0.3 else "red"
        plat_summary.add_row(
            platform,
            str(stats["ok"]), str(stats["err"]), str(stats["na"]),
            str(total),
            Text(f"{rate*100:5.1f}%", style=rate_style),
        )
    console.print(plat_summary)

    # ---- final headline ----
    console.print()
    headline = Text.assemble(
        ("Profile health: ", "bold"),
        (f"{n_full} FULL", "green"), ("  ·  ", "dim"),
        (f"{n_partial} PARTIAL", "yellow"), ("  ·  ", "dim"),
        (f"{n_dead} DEAD", "bold red"),
    )
    console.print(headline)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Probe every (profile × platform) combo for connectivity.",
    )
    parser.add_argument(
        "--registry", choices=("all", "generation", "construction"), default="all",
        help="which model registry to probe (default: all)",
    )
    parser.add_argument(
        "--profiles", type=_csv, default=None,
        help="optional comma-list filter (default: every profile in the chosen registry)",
    )
    parser.add_argument(
        "--only-failing", action="store_true",
        help="hide healthy rows in the per-combo table; per-profile + per-platform summaries unaffected",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    registries = ("construction", "generation") if args.registry == "all" else (args.registry,)
    return asyncio.run(main_async(
        registries=registries,
        profile_filter=args.profiles,
        only_failing=args.only_failing,
    ))


if __name__ == "__main__":
    raise SystemExit(main())
