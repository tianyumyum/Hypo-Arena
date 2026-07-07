"""Deep-probe a profile as a judge candidate: connectivity + thinking-mode verification.

Usage:
    uv run python scripts/judge_probe.py                      # default: mimo-v2-flash, mimo-v2-pro
    uv run python scripts/judge_probe.py --profiles foo,bar

For each (profile, platform) combo, sends a simple prompt with the profile's configured
model_settings (including extra_body={"thinking": {"type": "disabled"}} when set), then
dumps the raw response so we can confirm:
  - the call succeeds end-to-end,
  - no reasoning_content / thinking_text leaks back,
  - reasoning_tokens in usage stays at 0.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Any

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.text import Text

from basics.models import GENERATION_REGISTRY
from basics.platform import get_client, is_available


DEFAULT_PROFILES = ("mimo-v2-flash", "mimo-v2-pro")

# Two-step prompt: simple PONG (probe whether thinking mode kicks in even on trivial input).
TEST_PROMPT = "Reply with exactly one word: PONG"
TIMEOUT_SEC = 60.0
MAX_TOKENS = 64


def _csv(value: str) -> tuple[str, ...]:
    return tuple(s.strip() for s in value.split(",") if s.strip())


def _safe_get(obj: Any, *keys: str) -> Any:
    """Navigate nested attributes/dict-keys safely; return None if any step missing."""
    for k in keys:
        if obj is None:
            return None
        obj = getattr(obj, k, None) if not isinstance(obj, dict) else obj.get(k)
    return obj


def _detect_thinking(message: Any, usage: Any) -> tuple[bool, list[str]]:
    """Return (thinking_was_emitted, list_of_signals)."""
    signals: list[str] = []
    # Common reasoning fields across vendors.
    for field in ("reasoning_content", "reasoning", "thinking", "thinking_text", "thought"):
        val = getattr(message, field, None) if not isinstance(message, dict) else message.get(field)
        if val:
            signals.append(f"message.{field} = {str(val)[:120]!r}")
    # OpenAI-style: usage.completion_tokens_details.reasoning_tokens
    rt = _safe_get(usage, "completion_tokens_details", "reasoning_tokens")
    if rt:
        signals.append(f"usage.completion_tokens_details.reasoning_tokens = {rt}")
    # Some vendors split: usage.reasoning_tokens
    rt2 = _safe_get(usage, "reasoning_tokens")
    if rt2:
        signals.append(f"usage.reasoning_tokens = {rt2}")
    return (bool(signals), signals)


async def probe_one(profile_name: str, platform: str, model_id: str, extra_body: dict | None) -> dict:
    """Send one request and return a structured probe result."""
    if not is_available(platform):
        return {"profile": profile_name, "platform": platform, "model_id": model_id,
                "status": "na", "detail": "env vars missing"}

    client = get_client(platform)
    t0 = time.time()
    try:
        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "max_tokens": MAX_TOKENS,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = await asyncio.wait_for(
            client.chat.completions.create(**kwargs),
            timeout=TIMEOUT_SEC,
        )
        elapsed = time.time() - t0

        msg = resp.choices[0].message
        text = (msg.content or "").strip()
        usage = resp.usage
        in_tokens = getattr(usage, "prompt_tokens", None) or getattr(usage, "input_tokens", None)
        out_tokens = getattr(usage, "completion_tokens", None) or getattr(usage, "output_tokens", None)

        thinking_emitted, signals = _detect_thinking(msg, usage)
        return {
            "profile": profile_name, "platform": platform, "model_id": model_id,
            "status": "ok",
            "elapsed": elapsed,
            "text": text,
            "in_tokens": in_tokens,
            "out_tokens": out_tokens,
            "thinking_emitted": thinking_emitted,
            "thinking_signals": signals,
        }
    except asyncio.TimeoutError:
        return {"profile": profile_name, "platform": platform, "model_id": model_id,
                "status": "err", "detail": f"TimeoutError: >{TIMEOUT_SEC}s"}
    except Exception as exc:
        return {"profile": profile_name, "platform": platform, "model_id": model_id,
                "status": "err", "detail": f"{type(exc).__name__}: {str(exc)[:200]}"}


async def main_async(profile_names: tuple[str, ...]) -> int:
    console = Console()

    triples: list[tuple[str, str, str, dict | None]] = []
    for name in profile_names:
        if name not in GENERATION_REGISTRY:
            console.print(f"[red]unknown profile: {name}[/red]")
            continue
        profile = GENERATION_REGISTRY[name]
        extra = profile.model_settings.extra_body
        for platform, model_id in profile.platform_ids:
            if model_id is None:
                continue
            triples.append((name, platform, model_id, extra))

    if not triples:
        console.print("[red]no probes to run[/red]")
        return 1

    console.print(
        f"Probing [bold cyan]{len(triples)}[/bold cyan] (profile, platform) combos.\n"
        f"Per-call timeout {TIMEOUT_SEC:.0f}s. Sending: {TEST_PROMPT!r}\n"
    )
    # Show extra_body once per profile so user can see what's being sent.
    seen: set[str] = set()
    for name, _, _, extra in triples:
        if name in seen:
            continue
        seen.add(name)
        console.print(f"  [dim]{name} extra_body =[/dim] [white]{json.dumps(extra)}[/white]")
    console.print()

    sem = asyncio.Semaphore(8)

    async def wrapped(name, platform, model_id, extra):
        async with sem:
            return await probe_one(name, platform, model_id, extra)

    results = await asyncio.gather(
        *(wrapped(n, p, m, e) for n, p, m, e in triples)
    )

    # Render compact table
    table = Table(title="Judge Candidate Probe", title_style="bold cyan", expand=False)
    table.add_column("Profile", no_wrap=True, style="white", min_width=15)
    table.add_column("Platform", no_wrap=True, min_width=8)
    table.add_column("Model ID", no_wrap=True, min_width=15, overflow="ellipsis", max_width=22)
    table.add_column("Status", no_wrap=True, min_width=6)
    table.add_column("Elapsed", justify="right", no_wrap=True, min_width=7)
    table.add_column("In/Out tokens", justify="right", no_wrap=True, min_width=12)
    table.add_column("Thinking?", no_wrap=True, min_width=11)
    table.add_column("Reply / Error", overflow="ellipsis", max_width=40, no_wrap=True)

    style_by_status = {"ok": "green", "err": "red", "na": "yellow"}
    sym_by_status = {"ok": "✓", "err": "✗", "na": "·"}

    current_profile = None
    for r in results:
        if r["profile"] != current_profile:
            if current_profile is not None:
                table.add_section()
            current_profile = r["profile"]
        if r["status"] == "ok":
            thinking_cell = (
                Text("⚠ EMITTED", style="bold red") if r["thinking_emitted"]
                else Text("disabled ✓", style="green")
            )
            table.add_row(
                r["profile"], r["platform"], r["model_id"],
                Text(f"{sym_by_status['ok']} OK", style="green"),
                f"{r['elapsed']:.1f}s",
                f"{r['in_tokens']}/{r['out_tokens']}",
                thinking_cell,
                Text(repr(r["text"]), style="green"),
            )
        else:
            table.add_row(
                r["profile"], r["platform"], r["model_id"],
                Text(f"{sym_by_status[r['status']]} {r['status'].upper()}", style=style_by_status[r["status"]]),
                "-", "-", "-",
                Text(r["detail"], style=style_by_status[r["status"]]),
            )
    console.print(table)

    # Print thinking-signal details for any combo that leaked
    any_leak = False
    for r in results:
        if r["status"] == "ok" and r["thinking_emitted"]:
            if not any_leak:
                console.print("\n[bold red]Thinking-mode leak details:[/bold red]")
                any_leak = True
            console.print(f"  [yellow]{r['profile']} @ {r['platform']}:[/yellow]")
            for sig in r["thinking_signals"]:
                console.print(f"    - {sig}")

    # Print full error details (since the table column is truncated)
    err_results = [r for r in results if r["status"] == "err"]
    if err_results:
        console.print("\n[bold red]Error details:[/bold red]")
        for r in err_results:
            console.print(f"  [yellow]{r['profile']} @ {r['platform']} ({r['model_id']}):[/yellow]")
            console.print(f"    {r['detail']}")

    # Per-profile summary
    console.print()
    summary = Table(title="Judge Candidate Summary", title_style="bold cyan", expand=False)
    summary.add_column("Profile", no_wrap=True)
    summary.add_column("Healthy", justify="right")
    summary.add_column("Thinking truly disabled?", no_wrap=True)
    summary.add_column("Verdict", no_wrap=True)
    for name in profile_names:
        if name not in GENERATION_REGISTRY:
            continue
        prof_results = [r for r in results if r["profile"] == name]
        ok = [r for r in prof_results if r["status"] == "ok"]
        leaks = [r for r in ok if r["thinking_emitted"]]
        healthy_str = f"{len(ok)} / {len(prof_results)}"
        if not ok:
            verdict = Text("UNUSABLE (no platform works)", style="bold red")
            think_str = Text("n/a", style="yellow")
        elif leaks:
            verdict = Text("USABLE but thinking leaks on some platform", style="bold yellow")
            think_str = Text(f"NO ({len(leaks)} platform(s) leak)", style="red")
        else:
            verdict = Text("USABLE & thinking off", style="bold green")
            think_str = Text("YES", style="green")
        summary.add_row(name, healthy_str, think_str, verdict)
    console.print(summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe a profile as a judge candidate (connectivity + thinking).")
    parser.add_argument(
        "--profiles", type=_csv, default=",".join(DEFAULT_PROFILES),
        help="comma-list of profile names (default: mimo-v2-flash, mimo-v2-pro)",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return asyncio.run(main_async(args.profiles))


if __name__ == "__main__":
    raise SystemExit(main())
