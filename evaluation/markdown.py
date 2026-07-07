"""Markdown rendering for Leaderboard — companion of the JSON serialization.

Both arena and score leaderboards share `basics.schema.Leaderboard`, so a single
renderer handles both. Output is a header block (config / judge / counts / timestamp)
followed by a ranked table whose breakdown columns are inferred from the first
ranking entry — arena breakdowns are win/loss/tie, score breakdowns are rubric
dimensions (grounding, insight, ..., S, recall as applicable).
"""

from __future__ import annotations

from basics import Leaderboard


def _fmt_num(value, *, digits: int = 2) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        if value.is_integer():                                  # win/loss/tie etc. read cleaner without .00
            return str(int(value))
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_int(value) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def render_leaderboard_md(lb: Leaderboard) -> str:
    """Render a Leaderboard as a markdown document with header and ranked table."""
    md = lb.metadata
    title_scope = md.domain or (
        "cross-domain (" + ", ".join(md.domains or []) + ")" if md.domains else "cross-domain"
    )

    extras: list[str] = []
    if md.position_consistency_rate is not None:
        extras.append(f"**Position consistency**: {md.position_consistency_rate:.1%}")
    if md.btd_iterations:
        extras.append(f"**BTD iters**: {md.btd_iterations}")
    extras_line = "  ·  ".join(extras)

    lines: list[str] = [
        f"# {md.method.title()} Leaderboard — {title_scope}",
        "",
        f"**Config**: `{md.config}`  ·  **Judge**: `{md.judge}`  ·  **Method**: `{md.method}`",
        f"**Models**: {md.n_models}  ·  **Observations**: {_fmt_int(md.n_observations)}",
    ]
    if extras_line:
        lines.append(extras_line)
    lines.append(f"**Updated**: {md.created_at.isoformat(timespec='seconds')}")
    lines.append("")

    if not lb.rankings:
        lines.append("_(no rankings yet)_")
        lines.append("")
        return "\n".join(lines)

    # Use first row's breakdown keys for column ordering. Pad missing values with —.
    breakdown_keys = list(lb.rankings[0].breakdown.keys())
    headers = ["Rank", "Model", "Rating", *breakdown_keys, "n"]
    aligns = ["---:", ":---", "---:"] + ["---:"] * len(breakdown_keys) + ["---:"]

    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(aligns) + "|")

    for entry in lb.rankings:
        row = [
            str(entry.rank),
            f"`{entry.model}`",
            _fmt_num(entry.rating),
        ]
        for key in breakdown_keys:
            row.append(_fmt_num(entry.breakdown.get(key)))
        row.append(_fmt_int(entry.n_observations))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    return "\n".join(lines)
