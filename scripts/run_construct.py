"""Construction CLI: build BenchmarkCase JSONL from source metadata, streaming & resumable."""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

_HYPO_ROOT = Path(__file__).resolve().parents[1]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import configure_runtime
from basics import ALL_DOMAINS, SourceRecord, get_domain
from basics.io import append_case, existing_ids, load_sources
from basics.paths import cases_path
from construction import DEFAULT_MAX_ROUNDS, construct_case

logger = logging.getLogger("hypo.construct")


async def _construct_one(
    *,
    domain_name: str,
    max_rounds: int,
    profile: str,
    record: SourceRecord,
    semaphore: asyncio.Semaphore,
) -> None:
    domain = get_domain(domain_name)
    async with semaphore:
        try:
            case = await construct_case(
                domain=domain,
                forge_profile=profile,
                max_rounds=max_rounds,
                record=record,
            )
        except Exception as exc:
            logger.exception("construct.fail id=%s err=%s", record.id, exc)
            return
        append_case(domain_name, profile, case)
        logger.info(
            "construct.ok id=%s passed=%s",
            case.id, case.quality.passed,
        )


async def run(
    *,
    concurrency: int,
    domains: list[str],
    limit: int | None,
    max_rounds: int,
    profile: str,
) -> None:
    """Construct every pending source record across the requested domains."""
    for domain_name in domains:
        sources = load_sources(domain_name)
        if limit is not None:
            sources = sources[:limit]
        done = existing_ids(cases_path(domain_name, profile))
        pending = [r for r in sources if r.id not in done]
        logger.info(
            "construct.start domain=%s profile=%s total=%d pending=%d",
            domain_name, profile, len(sources), len(pending),
        )
        if not pending:
            continue
        semaphore = asyncio.Semaphore(concurrency)
        await asyncio.gather(
            *(
                _construct_one(
                    domain_name=domain_name,
                    max_rounds=max_rounds,
                    profile=profile,
                    record=record,
                    semaphore=semaphore,
                )
                for record in pending
            )
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HypoArena construction pipeline")
    parser.add_argument("--domain", action="append", choices=list(ALL_DOMAINS), required=True)
    parser.add_argument("--profile", default="gpt-5.4-high")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-rounds", type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    configure_runtime()
    asyncio.run(
        run(
            concurrency=args.concurrency,
            domains=args.domain,
            limit=args.limit,
            max_rounds=args.max_rounds,
            profile=args.profile,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
