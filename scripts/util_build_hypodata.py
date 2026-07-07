"""Aggregate per-domain cases into the anonymized release dataset hypodata.json.

Reads ``artifacts/{domain}/cases/{config}.jsonl`` for every domain, strips the
runtime-only fields (provenance / quality / schema_version), normalizes the
metadata shape per domain to match the paper-submitted release schema, fixes
key order, and writes a single compact JSON array.

Usage:
  uv run python -m scripts.util_build_hypodata [--config gpt-5.4] \\
      [--output path/to/hypodata.json] [--no-backup]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import ALL_DOMAINS
from basics.paths import cases_path

logger = logging.getLogger("hypo.util_build_hypodata")

DEFAULT_CONFIG = "gpt-5.4"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "hypodata.json"


def _project_metadata(domain: str, meta: dict) -> dict:
    """Map per-domain metadata to the anonymized release schema."""
    if domain == "it_operations":
        case_id = meta.get("report_id", "")
        return {"case_id": case_id, "title": case_id}
    if domain == "safety_investigation":
        case_id = meta.get("report_id", "")
        return {
            "source": meta.get("agency", ""),
            "case_id": case_id,
            "title": case_id,
        }
    return dict(meta)


def _project_hypothesis(h: dict) -> dict:
    return {
        "hypothesis": h.get("hypothesis", ""),
        "evidence": h.get("evidence", ""),
        "category": h.get("category"),
    }


def _project_case(row: dict) -> dict:
    return {
        "id": row["id"],
        "domain": row["domain"],
        "metadata": _project_metadata(row["domain"], row.get("metadata", {})),
        "context": row.get("context", ""),
        "hypotheses": [_project_hypothesis(h) for h in row.get("hypotheses", [])],
    }


def _load_domain(domain: str, config: str) -> list[dict]:
    path = cases_path(domain, config)
    if not path.exists():
        logger.warning("missing %s; skipping domain", path)
        return []
    cases: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            cases.append(_project_case(json.loads(line)))
    logger.info("loaded %s: %d cases", domain, len(cases))
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG,
                        help=f"construction config to read (default: {DEFAULT_CONFIG})")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help=f"target file path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--no-backup", action="store_true",
                        help="overwrite output without creating .bak.<timestamp>")
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname).1s] %(name)s: %(message)s",
                        datefmt="%H:%M:%S")

    aggregated: list[dict] = []
    for domain in ALL_DOMAINS:
        aggregated.extend(_load_domain(domain, args.config))
    aggregated.sort(key=lambda r: (r["domain"], r["id"]))

    if args.output.exists() and not args.no_backup:
        backup = args.output.with_suffix(
            args.output.suffix + f".bak.{datetime.now():%Y%m%d-%H%M%S}",
        )
        args.output.replace(backup)
        logger.info("backed up prior file to %s", backup)

    args.output.write_text(
        json.dumps(aggregated, ensure_ascii=False, separators=(", ", ": ")),
        encoding="utf-8",
    )
    n_hyps = sum(len(c["hypotheses"]) for c in aggregated)
    print(f"wrote {len(aggregated)} cases ({n_hyps} hypotheses) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
