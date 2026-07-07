"""Generation CLI: produce Submissions for one (domain, construction profile, model, mode)."""

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
from basics import ALL_DOMAINS, BenchmarkCase, get_domain
from basics.io import append_submission, existing_ids, load_cases
from basics.paths import submission_path
from generation import generate_submission

logger = logging.getLogger("hypo.generate")


async def _generate_one(
    *,
    case: BenchmarkCase,
    construction_profile: str,
    domain_name: str,
    mode: str,
    profile: str,
    semaphore: asyncio.Semaphore,
) -> None:
    domain = get_domain(domain_name)
    async with semaphore:
        try:
            submission = await generate_submission(
                case=case,
                domain=domain,
                mode=mode,
                profile_name=profile,
            )
        except Exception as exc:
            logger.exception("generate.fail id=%s err=%s", case.id, exc)
            return
        append_submission(domain_name, mode, construction_profile, profile, submission)
        logger.info(
            "generate.ok id=%s mode=%s profile=%s skills=%s",
            case.id,
            mode,
            profile,
            ",".join(submission.provenance.skills_used or []) or "-",
        )


async def run(
    *,
    concurrency: int,
    construction_profile: str,
    domain_name: str,
    limit: int | None,
    mode: str,
    profile: str,
) -> None:
    """Generate submissions for every passed BenchmarkCase under (domain, construction_profile)."""
    cases = load_cases(domain_name, construction_profile, only_passed=True)
    if limit is not None:
        cases = cases[:limit]
    done = existing_ids(submission_path(domain_name, mode, construction_profile, profile))
    pending = [c for c in cases if c.id not in done]
    logger.info(
        "generate.start domain=%s mode=%s config=%s profile=%s total=%d pending=%d",
        domain_name, mode, construction_profile, profile, len(cases), len(pending),
    )
    if not pending:
        return
    semaphore = asyncio.Semaphore(concurrency)
    await asyncio.gather(
        *(
            _generate_one(
                case=case,
                construction_profile=construction_profile,
                domain_name=domain_name,
                mode=mode,
                profile=profile,
                semaphore=semaphore,
            )
            for case in pending
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="HypoArena generation pipeline")
    parser.add_argument("--domain", choices=list(ALL_DOMAINS), required=True)
    parser.add_argument("--construction-profile", default="gpt-5.4-high")
    parser.add_argument("--profile", required=True, help="Generation profile from basics.GENERATION_REGISTRY")
    parser.add_argument("--mode", choices=["baseline", "agent"], default="baseline")
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    configure_runtime()
    asyncio.run(
        run(
            concurrency=args.concurrency,
            construction_profile=args.construction_profile,
            domain_name=args.domain,
            limit=args.limit,
            mode=args.mode,
            profile=args.profile,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
