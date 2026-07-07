"""Aggregate Agent-Mode skill-selection statistics from existing submissions.

Walks every `artifacts/{domain}/submissions/agent/*.jsonl`, extracts the
`provenance.skills_used` from each row, and reports:

  1. Per-skill total invocation counts (with share %).
  2. Pipeline-length distribution (how often agents pick 1 / 2 / 3 skills).
  3. Per-skill share within each domain (which skills each domain prefers).
  4. Per-skill share within each generation profile (which models prefer
     which skills).
  5. Top skill pairs and triples (which combinations co-occur).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from rich.console import Console
from rich.table import Table
from rich.text import Text

from basics import ALL_DOMAINS
from basics.paths import ARTIFACTS_ROOT
from generation.skills import SKILL_NAMES


def _iter_agent_submissions():
    """Yield (domain, profile, skills_used) for every agent-mode submission row."""
    for domain in ALL_DOMAINS:
        agent_dir = ARTIFACTS_ROOT / domain / "submissions" / "agent"
        if not agent_dir.exists():
            continue
        for jsonl in sorted(agent_dir.glob("*.jsonl")):
            stem = jsonl.stem                                  # "{config}+{profile}"
            profile = stem.split("+", 1)[1] if "+" in stem else stem
            with jsonl.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    skills = row.get("provenance", {}).get("skills_used") or []
                    yield domain, profile, list(skills)


def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "  -  "
    return f"{num / denom * 100:5.1f}%"


def _sortable_skill_index(name: str) -> int:
    try:
        return SKILL_NAMES.index(name)
    except ValueError:
        return len(SKILL_NAMES)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate Agent-Mode skill invocation statistics from artifacts/."
    )
    parser.add_argument("--domains", default="all",
                        help="comma-list (default: all 6)")
    parser.add_argument("--top-pairs", type=int, default=10,
                        help="how many top skill pairs to show (default 10)")
    parser.add_argument("--top-triples", type=int, default=8,
                        help="how many top skill triples to show (default 8)")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    selected_domains = (
        tuple(ALL_DOMAINS) if args.domains.strip().lower() == "all"
        else tuple(s.strip() for s in args.domains.split(",") if s.strip())
    )

    rows = [
        (d, p, s)
        for d, p, s in _iter_agent_submissions()
        if d in selected_domains
    ]

    console = Console()
    if not rows:
        console.print("[red]no agent-mode submissions found[/red]")
        return 1

    n_subs = len(rows)
    n_with_skills = sum(1 for _, _, s in rows if s)
    n_total_invocations = sum(len(s) for _, _, s in rows)

    console.print(
        f"Scanned [bold cyan]{n_subs}[/bold cyan] agent-mode submissions across "
        f"[bold cyan]{len(selected_domains)}[/bold cyan] domain(s); "
        f"[bold cyan]{n_with_skills}[/bold cyan] carry a non-empty skill pipeline; "
        f"[bold cyan]{n_total_invocations}[/bold cyan] total skill invocations.\n"
    )

    # ---- 1. per-skill totals ----
    per_skill = Counter(s for _, _, skills in rows for s in skills)
    table1 = Table(title="Per-skill invocation totals", title_style="bold cyan", expand=False)
    table1.add_column("Rank", justify="right", no_wrap=True)
    table1.add_column("Skill", no_wrap=True)
    table1.add_column("Invocations", justify="right")
    table1.add_column("Share of all calls", justify="right")
    table1.add_column("Submissions using it", justify="right")
    table1.add_column("Share of submissions", justify="right")

    submission_uses_skill: Counter[str] = Counter()
    for _, _, skills in rows:
        for s in set(skills):                                  # count once per submission
            submission_uses_skill[s] += 1

    ranked = per_skill.most_common()
    for rank, (skill, count) in enumerate(ranked, 1):
        sub_count = submission_uses_skill[skill]
        table1.add_row(
            str(rank), skill,
            str(count), _pct(count, n_total_invocations),
            str(sub_count), _pct(sub_count, n_subs),
        )
    # Skills that are never picked
    never = sorted(set(SKILL_NAMES) - set(per_skill), key=_sortable_skill_index)
    for skill in never:
        table1.add_row("—", Text(skill, style="dim"), "0", _pct(0, 1), "0", _pct(0, 1))
    console.print(table1)

    # ---- 2. pipeline-length distribution ----
    length_dist = Counter(len(s) for _, _, s in rows)
    table2 = Table(title="Pipeline length (skills per submission)", title_style="bold cyan", expand=False)
    table2.add_column("Length", justify="right")
    table2.add_column("Submissions", justify="right")
    table2.add_column("Share", justify="right")
    for length in sorted(length_dist):
        table2.add_row(str(length), str(length_dist[length]), _pct(length_dist[length], n_subs))
    console.print()
    console.print(table2)

    # ---- 3. skill × domain matrix (skills as rows for narrow-terminal fit) ----
    domain_skill_counts: dict[str, Counter[str]] = {d: Counter() for d in selected_domains}
    domain_totals: Counter[str] = Counter()
    for d, _, skills in rows:
        for s in skills:
            domain_skill_counts[d][s] += 1
            domain_totals[d] += 1

    table3 = Table(title="Skill × Domain matrix (% of that domain's invocations)",
                   title_style="bold cyan", expand=False)
    table3.add_column("Skill", no_wrap=True, style="white")
    for d in selected_domains:
        table3.add_column(d, justify="right", no_wrap=True)
    skills_in_use = [s for s, _ in ranked]
    for s in skills_in_use:
        cell = [s]
        for d in selected_domains:
            v = domain_skill_counts[d][s]
            cell.append(_pct(v, domain_totals[d]) if v else "  ·  ")
        table3.add_row(*cell)
    table3.add_row(
        Text("(domain total invocations)", style="dim"),
        *[Text(str(domain_totals[d]), style="bold") for d in selected_domains],
    )
    console.print()
    console.print(table3)

    # ---- 4. skill × profile matrix (skills as rows, profiles as columns) ----
    profile_skill_counts: dict[str, Counter[str]] = {}
    profile_totals: Counter[str] = Counter()
    for _, p, skills in rows:
        profile_skill_counts.setdefault(p, Counter())
        for s in skills:
            profile_skill_counts[p][s] += 1
            profile_totals[p] += 1

    profiles_sorted = sorted(profile_totals, key=lambda x: -profile_totals[x])

    table4 = Table(title="Skill × Profile matrix (% of that profile's invocations)",
                   title_style="bold cyan", expand=False)
    table4.add_column("Skill", no_wrap=True, style="white")
    # Abbreviate profile names so the table fits — keep enough to disambiguate
    def _abbr(p: str) -> str:
        # collapse "minimax-m2.7-thinking" → "minimax2.7", "claude-sonnet-4.6-high" → "claude-s4.6"
        return p.replace("-thinking", "").replace("-high", "").replace("-pro", "P").replace("-flash", "F")
    for p in profiles_sorted:
        table4.add_column(_abbr(p), justify="right", no_wrap=True)
    for s in skills_in_use:
        cell = [s]
        for p in profiles_sorted:
            v = profile_skill_counts[p][s]
            cell.append(_pct(v, profile_totals[p]) if v else "  ·  ")
        table4.add_row(*cell)
    table4.add_row(
        Text("(profile total invocations)", style="dim"),
        *[Text(str(profile_totals[p]), style="bold") for p in profiles_sorted],
    )
    console.print()
    console.print(table4)

    # ---- 5. top skill pairs / triples ----
    pair_counter: Counter[tuple[str, str]] = Counter()
    triple_counter: Counter[tuple[str, str, str]] = Counter()
    for _, _, skills in rows:
        unique = sorted(set(skills))
        for combo in combinations(unique, 2):
            pair_counter[combo] += 1
        if len(unique) >= 3:
            for combo in combinations(unique, 3):
                triple_counter[combo] += 1

    if pair_counter:
        table5 = Table(title=f"Top {args.top_pairs} skill pairs (co-occurrence count)",
                       title_style="bold cyan", expand=False)
        table5.add_column("Rank", justify="right")
        table5.add_column("Skill A", no_wrap=True)
        table5.add_column("Skill B", no_wrap=True)
        table5.add_column("Submissions", justify="right")
        for i, ((a, b), c) in enumerate(pair_counter.most_common(args.top_pairs), 1):
            table5.add_row(str(i), a, b, str(c))
        console.print()
        console.print(table5)

    if triple_counter:
        table6 = Table(title=f"Top {args.top_triples} skill triples",
                       title_style="bold cyan", expand=False)
        table6.add_column("Rank", justify="right")
        table6.add_column("Skill A", no_wrap=True)
        table6.add_column("Skill B", no_wrap=True)
        table6.add_column("Skill C", no_wrap=True)
        table6.add_column("Submissions", justify="right")
        for i, ((a, b, c), n) in enumerate(triple_counter.most_common(args.top_triples), 1):
            table6.add_row(str(i), a, b, c, str(n))
        console.print()
        console.print(table6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
