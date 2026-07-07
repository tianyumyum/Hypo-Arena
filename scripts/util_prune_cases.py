"""Drop bad cases from per-domain cases AND downstream submissions/results JSONL.

Reads a JSONL of bad-case fingerprints. Each entry is either:

  - a metadata dict (e.g. ``{"doi": "..."}`` or ``{"case_id": "...", "title": "..."}``)
    matched against ``case.metadata`` via the release-schema projection, or
  - ``{"id": "domain:xxx"}`` for a direct id match.

Resolves fingerprints to a set of case IDs by scanning live cases jsonl AND
any ``cases/*.bak.*`` files, so the script remains effective even after the
cases jsonl has already been pruned in a prior run.

For each affected domain (containing at least one matched ID), prunes:

  - ``cases/{config}.jsonl``                     (key: ``id``)
  - ``submissions/baseline/{config}+*.jsonl``    (key: ``id``)
  - ``submissions/agent/{config}+*.jsonl``       (key: ``id``)
  - ``results/{config}.*.arena.matches.jsonl``   (key: ``case_id``)
  - ``results/{config}.*.score.jsonl``           (key: ``case_id``)

Aggregated leaderboards (``results/*.leaderboard.{json,md}``) are left as-is —
re-run ``scripts.util_rebuild_leaderboards`` afterwards to refresh them, and
``scripts.util_build_hypodata`` to refresh the release dataset.

Usage:
  uv run python -m scripts.util_prune_cases [--bad-file hypo/bad_cases.jsonl] \\
      [--config gpt-5.4] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import ALL_DOMAINS
from basics.paths import ARTIFACTS_ROOT, cases_path, submissions_glob
from scripts.util_build_hypodata import _project_metadata

logger = logging.getLogger("hypo.util_prune_cases")

DEFAULT_CONFIG = "gpt-5.4"
DEFAULT_BAD_FILE = _HYPO_ROOT / "bad_cases.jsonl"


def _matches(fingerprint: dict, case: dict) -> bool:
    """True if every fingerprint key equals the case's id, domain, or projected metadata."""
    projected = _project_metadata(case["domain"], case.get("metadata", {}))
    for key, expected in fingerprint.items():
        if key == "id":
            if case.get("id") != expected:
                return False
        elif key == "domain":
            if case.get("domain") != expected:
                return False
        else:
            if projected.get(key) != expected:
                return False
    return True


def _load_fingerprints(path: Path) -> list[dict]:
    """Load non-empty fingerprint dicts; refuse the file if any line is ``{}``.

    An empty fingerprint would match every case via vacuous-truth on the matcher
    loop, which combined with the downstream prune would erase the entire
    benchmark — so we hard-fail rather than silently skip.
    """
    fingerprints: list[dict] = []
    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(),
                                  start=1):
        line = raw.strip()
        if not line:
            continue
        fp = json.loads(line)
        if not isinstance(fp, dict):
            raise ValueError(
                f"{path}:{lineno}: fingerprint must be a JSON object, got {type(fp).__name__}"
            )
        if not fp:
            raise ValueError(
                f"{path}:{lineno}: empty fingerprint {{}} would match every case; "
                "refusing to load. Add at least one identifying key (id / doi / case_id / ...)."
            )
        fingerprints.append(fp)
    return fingerprints


_ORCHESTRATOR_NEEDLES = (
    "run_orchestrate", "run_construct", "run_generate", "run_evaluate",
)


def _detect_orchestrators() -> list[tuple[str, str]]:
    """Return [(pid, command)] for processes that may currently write artifacts/."""
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid=,command="],
            capture_output=True, text=True, timeout=5, check=True,
        ).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return []
    hits: list[tuple[str, str]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip our own process.
        if "util_prune_cases" in line:
            continue
        if any(needle in line for needle in _ORCHESTRATOR_NEEDLES):
            pid, _, cmd = line.partition(" ")
            hits.append((pid.strip(), cmd.strip()))
    return hits


def _resolve_bad_ids(
    fingerprints: list[dict], config: str,
) -> tuple[dict[str, set[str]], list[set[str]]]:
    """Resolve fingerprints to bad case IDs by scanning live cases AND backups.

    Returns ``(by_domain, fp_match_ids)`` where:
      - ``by_domain[domain]`` is the set of case IDs to drop
      - ``fp_match_ids[i]`` is the set of distinct case IDs that fingerprint i hit,
        used downstream to verify each fingerprint was 1:1 specific.
    """
    by_domain: dict[str, set[str]] = {d: set() for d in ALL_DOMAINS}
    fp_match_ids: list[set[str]] = [set() for _ in fingerprints]

    def _scan(jsonl_path: Path, domain: str) -> None:
        if not jsonl_path.exists():
            return
        with jsonl_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                case = json.loads(line)
                cid = case.get("id")
                if not cid:
                    continue
                for i, fp in enumerate(fingerprints):
                    if _matches(fp, case):
                        by_domain[domain].add(cid)
                        fp_match_ids[i].add(cid)

    for domain in ALL_DOMAINS:
        live = cases_path(domain, config)
        _scan(live, domain)
        cases_dir = live.parent
        if cases_dir.exists():
            for bak in sorted(cases_dir.glob(f"{live.name}.bak.*")):
                _scan(bak, domain)
    return by_domain, fp_match_ids


def _prune_jsonl_by_id(path: Path, *, id_key: str, bad_ids: set[str],
                       dry_run: bool) -> int:
    """Drop rows whose id_key value is in bad_ids; back up + rewrite if changed."""
    if not path.exists():
        return 0
    kept: list[str] = []
    dropped = 0
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get(id_key) in bad_ids:
                dropped += 1
                continue
            kept.append(json.dumps(row, ensure_ascii=False))
    if dropped == 0:
        return 0
    if dry_run:
        return dropped
    backup = path.with_suffix(
        path.suffix + f".bak.{datetime.now():%Y%m%d-%H%M%S}",
    )
    path.replace(backup)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dropped


def _prune_domain(domain: str, *, config: str, bad_ids: set[str],
                  dry_run: bool) -> dict[str, int]:
    """Prune cases + submissions + results for one domain. Returns counts by file group."""
    counts = {"cases": 0, "submissions": 0, "arena_matches": 0, "scores": 0}
    if not bad_ids:
        return counts

    counts["cases"] = _prune_jsonl_by_id(
        cases_path(domain, config), id_key="id",
        bad_ids=bad_ids, dry_run=dry_run,
    )
    for mode in ("baseline", "agent"):
        for sub in submissions_glob(domain, mode, config):
            counts["submissions"] += _prune_jsonl_by_id(
                sub, id_key="id", bad_ids=bad_ids, dry_run=dry_run,
            )
    results_dir = ARTIFACTS_ROOT / domain / "results"
    if results_dir.exists():
        for arena in results_dir.glob(f"{config}.*.arena.matches.jsonl"):
            counts["arena_matches"] += _prune_jsonl_by_id(
                arena, id_key="case_id", bad_ids=bad_ids, dry_run=dry_run,
            )
        for score in results_dir.glob(f"{config}.*.score.jsonl"):
            counts["scores"] += _prune_jsonl_by_id(
                score, id_key="case_id", bad_ids=bad_ids, dry_run=dry_run,
            )
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bad-file", type=Path, default=DEFAULT_BAD_FILE,
                        help=f"jsonl of bad-case fingerprints (default: {DEFAULT_BAD_FILE})")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true",
                        help="report drop counts without rewriting any jsonl")
    parser.add_argument("--force", action="store_true",
                        help="proceed even if an orchestrator/generation process appears active")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    fingerprints = _load_fingerprints(args.bad_file)
    logger.info("loaded %d fingerprints from %s", len(fingerprints), args.bad_file)

    bad_by_domain, fp_match_ids = _resolve_bad_ids(fingerprints, args.config)
    total_ids = sum(len(s) for s in bad_by_domain.values())
    logger.info("resolved %d unique case id(s) across %d domain(s)",
                total_ids, sum(1 for s in bad_by_domain.values() if s))

    # Per-fingerprint match audit so the user can spot ambiguous matches.
    print("\nfingerprint match audit:")
    ambiguous_or_missed = False
    for fp, hits in zip(fingerprints, fp_match_ids):
        n = len(hits)
        if n == 1:
            mark = " "
        elif n == 0:
            mark = "?"
            ambiguous_or_missed = True
        else:
            mark = "!"
            ambiguous_or_missed = True
        print(f"  [{mark}] {n:>2} hit(s) ← {json.dumps(fp, ensure_ascii=False)}")
        if n > 1:
            for cid in sorted(hits):
                print(f"        ↳ {cid}")
    if ambiguous_or_missed:
        print("  legend: [ ]=1:1 ok, [?]=no match (typo? already removed?), "
              "[!]=multiple matches (over-broad fingerprint?)")

    if total_ids == 0:
        print("\nnothing to prune (no fingerprints matched any cases or backups)")
        return 0

    # Orchestrator detection — concurrent writers could re-introduce data we delete.
    orch = _detect_orchestrators()
    if orch:
        print("\n!! WARNING: detected possibly-running orchestrator/generation process(es):")
        for pid, cmd in orch:
            print(f"  pid {pid}: {cmd[:160]}")
        if not args.dry_run and not args.force:
            print("\nRefusing to apply while these processes are active. "
                  "Stop them or re-run with --force to override.")
            return 2
        if not args.dry_run:
            print("  (--force given; proceeding anyway)")

    grand: dict[str, int] = {"cases": 0, "submissions": 0,
                              "arena_matches": 0, "scores": 0}
    affected_domains = 0
    for domain in ALL_DOMAINS:
        ids = bad_by_domain[domain]
        if not ids:
            continue
        affected_domains += 1
        counts = _prune_domain(domain, config=args.config,
                                bad_ids=ids, dry_run=args.dry_run)
        marker = " (dry-run)" if args.dry_run else ""
        print(f"\n[{domain}] {len(ids)} bad id(s){marker}")
        for cid in sorted(ids):
            print(f"    - {cid}")
        print(f"    cases:{counts['cases']}  submissions:{counts['submissions']}  "
              f"arena_matches:{counts['arena_matches']}  scores:{counts['scores']}")
        for k, v in counts.items():
            grand[k] += v

    print(f"\nTotal across {affected_domains} domain(s){' (dry-run)' if args.dry_run else ''}: "
          f"cases={grand['cases']}, submissions={grand['submissions']}, "
          f"arena_matches={grand['arena_matches']}, scores={grand['scores']}")
    if not args.dry_run and (grand["arena_matches"] or grand["scores"]):
        print("Next: re-run `uv run python -m scripts.util_rebuild_leaderboards` "
              "to refresh aggregated leaderboards, then "
              "`uv run python -m scripts.util_build_hypodata` to refresh hypodata.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
