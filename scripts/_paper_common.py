"""Shared helpers for paper artifact builders (tables + figures).

All paper-stage scripts read finished leaderboard JSONs (and optionally raw
arena.matches.jsonl / score.jsonl) under `artifacts/{domain}/results/`. This
module centralises three concerns:

  * **Domain ordering & display labels** — paper renders Research before
    Real-World, with abbreviated column headers.
  * **Model-label normalization** — artifacts store labels like
    `baseline:claude-sonnet-4.6-high`; the paper drops the mode prefix and the
    `-high` / `-thinking` suffixes.
  * **Leaderboard loading** — small dataclass that flattens a
    `Leaderboard` JSON into rows ready for tabulation.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_HYPO_ROOT = Path(__file__).resolve().parents[2]
if str(_HYPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_HYPO_ROOT))

from basics import Pool
from basics.paths import (
    ARTIFACTS_ROOT,
    arena_pool_leaderboard_path,
    score_pool_leaderboard_path,
)

# ---- domain ordering & labels --------------------------------------------

RESEARCH_DOMAINS: tuple[str, ...] = (
    "biomedical_science",
    "machine_learning",
    "social_science",
)
REAL_WORLD_DOMAINS: tuple[str, ...] = (
    "financial_analysis",
    "it_operations",
    "safety_investigation",
)
ORDERED_DOMAINS: tuple[str, ...] = RESEARCH_DOMAINS + REAL_WORLD_DOMAINS

DOMAIN_SHORT_LABEL: dict[str, str] = {
    "biomedical_science": "Bio",
    "machine_learning": "ML",
    "social_science": "Social",
    "financial_analysis": "Fin",
    "it_operations": "IT",
    "safety_investigation": "Safety",
}

DOMAIN_FULL_LABEL: dict[str, str] = {
    "biomedical_science": "Biomedical Science",
    "machine_learning": "Machine Learning",
    "social_science": "Social Science",
    "financial_analysis": "Financial Analysis",
    "it_operations": "IT Operations",
    "safety_investigation": "Safety Investigation",
}

# Default judge for paper headline numbers.
PRIMARY_JUDGE = "seed-2.0-pro"
CONSTRUCTION_PROFILE = "gpt-5.4"

# Both judges run in parallel under orchestrate.bash; paper builders may want to
# emit one artifact per judge. The short-name mapping keeps filenames clean —
# `arena-mimo.tex` reads better than `arena-mimo-v2-pro.tex`.
PAPER_JUDGES: tuple[str, ...] = ("mimo-v2-pro", "seed-2.0-pro")
JUDGE_SHORT_NAMES: dict[str, str] = {
    "mimo-v2-pro": "mimo",
    "mimo-v2-flash": "mimo-flash",
    "seed-2.0-pro": "seed",
}


def judge_short_name(judge: str) -> str:
    """Compact filename-friendly alias for a judge profile (defaults to first dash-segment)."""
    if judge in JUDGE_SHORT_NAMES:
        return JUDGE_SHORT_NAMES[judge]
    return judge.split("-", 1)[0]

# ---- paper-level model selection & analysis knobs ------------------------

# Baseline profiles that the paper's §4.2 arena table reports.
# Explicit whitelist so a fleet expansion doesn't silently add rows to Table 1.
PAPER_MAIN_TABLE_MODELS: tuple[str, ...] = (
    "claude-sonnet-4.6-high",
    "claude-opus-4.6-high",
    "deepseek-v4-flash-high",
    "deepseek-v4-pro-high",
    "gemini-3-flash-high",
    "gemini-3.1-pro-high",
    "glm-5-thinking",
    "glm-5.1-thinking",
    "gpt-5.4-mini-high",
    "gpt-5.4-high",
    "kimi-k2.5-thinking",
    "kimi-k2.6-thinking",
    "minimax-m2.5-thinking",
    "minimax-m2.7-thinking",
    "qwen-3.6-max-thinking",
)

# Profiles excluded from the §4.4 Agent-vs-Baseline figure. Aligned with
# PAPER_MAIN_TABLE_MODELS so the figure and Table 1 share a model set.
# Empty now that qwen-3.6-plus has been removed from the system entirely.
PAPER_AGENT_FIG_EXCLUDE: tuple[str, ...] = ()

# Capability tiers used by §4.4 (effect of skills). Thresholds are on baseline
# avg BTD rating; keep the boundary values here so downstream scripts and the
# paper prose stay in sync.
TIER_THRESHOLDS: dict[str, tuple[float, float]] = {
    # (inclusive lower, exclusive upper) — open upper/lower via +/-inf.
    "top":  (1560.0, float("inf")),
    "mid":  (1470.0, 1560.0),
    "low":  (float("-inf"), 1470.0),
}

# ---- model-label normalization -------------------------------------------

REFERENCE_LABEL = "reference"

_STRIP_SUFFIXES: tuple[str, ...] = ("-high", "-thinking")


def split_mode_label(raw_label: str) -> tuple[str, str]:
    """Split `baseline:claude-sonnet-4.6-high` → ("baseline", "claude-sonnet-4.6-high").

    Reference rows have no mode prefix and are returned with mode="reference".
    """
    if raw_label == REFERENCE_LABEL:
        return ("reference", REFERENCE_LABEL)
    if ":" in raw_label:
        mode, profile = raw_label.split(":", 1)
        return (mode, profile)
    return ("baseline", raw_label)


def display_model_name(profile: str) -> str:
    """Drop `-high` / `-thinking` decoration so the paper sees a clean model id."""
    if profile == REFERENCE_LABEL:
        return "Reference"
    name = profile
    for suffix in _STRIP_SUFFIXES:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name


# ---- leaderboard loading -------------------------------------------------

@dataclass(frozen=True)
class LeaderboardRow:
    """One model entry, post-normalization, for tabulation."""

    breakdown: dict[str, float]
    mode: str                          # "baseline" | "agent" | "reference"
    n_observations: int
    profile: str                       # raw profile string (e.g. claude-sonnet-4.6-high)
    raw_label: str                     # exact `model` field from the JSON
    rating: float

    @property
    def display_name(self) -> str:
        return display_model_name(self.profile)

    @property
    def win_rate(self) -> float | None:
        win = self.breakdown.get("win")
        loss = self.breakdown.get("loss")
        tie = self.breakdown.get("tie")
        if win is None or loss is None or tie is None:
            return None
        denom = win + loss + tie
        return (win / denom) if denom > 0 else None


def load_arena_leaderboard(
    domain: str,
    *,
    judge: str = PRIMARY_JUDGE,
    config: str = CONSTRUCTION_PROFILE,
    pool: Pool = "baseline",
) -> list[LeaderboardRow]:
    """Load the pool-filtered arena leaderboard for ``(domain, config, judge)``.

    Default pool is ``baseline`` so paper §4.2 / §4.3 callers get the right
    artifact without having to specify; ``full`` is required for §4.2 figure,
    §4.4, and §4.5.
    """
    return _load_leaderboard(arena_pool_leaderboard_path(domain, config, judge, pool))


def load_score_leaderboard(
    domain: str,
    *,
    judge: str = PRIMARY_JUDGE,
    config: str = CONSTRUCTION_PROFILE,
    pool: Pool = "baseline",
) -> list[LeaderboardRow]:
    """Load the pool-filtered score leaderboard for ``(domain, config, judge)``."""
    return _load_leaderboard(score_pool_leaderboard_path(domain, config, judge, pool))


def _rows_from_leaderboard_object(lb) -> list[LeaderboardRow]:
    """Flatten an in-memory Leaderboard into LeaderboardRow list (mirrors _load_leaderboard)."""
    out: list[LeaderboardRow] = []
    for entry in lb.rankings:
        mode, profile = split_mode_label(entry.model)
        out.append(
            LeaderboardRow(
                breakdown=dict(entry.breakdown),
                mode=mode,
                n_observations=int(entry.n_observations),
                profile=profile,
                raw_label=entry.model,
                rating=float(entry.rating),
            )
        )
    return out


def load_restricted_arena_leaderboard(
    domain: str,
    *,
    allow_labels: set[str],
    judge: str = PRIMARY_JUDGE,
    config: str = CONSTRUCTION_PROFILE,
) -> list[LeaderboardRow]:
    """Recompute BTD + WR on matches whose both sides are in ``allow_labels``.

    Reads raw ``arena.matches.jsonl`` (not a pre-aggregated pool leaderboard), so the
    ratings reflect a universe of exactly the requested labels. Intended for paper-side
    uses where the display set must also be the competition set (e.g. §4.2 Table 1's
    14-model whitelist with Reference, to match a strict reading of the paper).

    Returns the same LeaderboardRow shape as ``load_arena_leaderboard`` so callers can
    reuse ``rows_by_profile`` and friends.
    """
    from basics import get_domain
    from basics.io import load_arena_matches
    from evaluation import build_arena_leaderboard

    matches = load_arena_matches(domain, config, judge)
    sub = [m for m in matches if m.model_a in allow_labels and m.model_b in allow_labels]
    if not sub:
        return []
    lb = build_arena_leaderboard(
        config=config, domain=get_domain(domain), judge_profile=judge, matches=sub,
    )
    return _rows_from_leaderboard_object(lb)


def _load_leaderboard(path: Path) -> list[LeaderboardRow]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out: list[LeaderboardRow] = []
    for entry in data["rankings"]:
        mode, profile = split_mode_label(entry["model"])
        out.append(
            LeaderboardRow(
                breakdown=dict(entry.get("breakdown", {})),
                mode=mode,
                n_observations=int(entry.get("n_observations", 0)),
                profile=profile,
                raw_label=entry["model"],
                rating=float(entry["rating"]),
            )
        )
    return out


# ---- per-mode pivot helpers ---------------------------------------------

def rows_by_profile(
    rows: Iterable[LeaderboardRow],
    *,
    mode: str,
) -> dict[str, LeaderboardRow]:
    """Index leaderboard rows by profile, restricted to `mode` (or "reference")."""
    out: dict[str, LeaderboardRow] = {}
    for row in rows:
        if row.mode != mode:
            continue
        out[row.profile] = row
    return out


# ---- baseline-vs-agent uplift helpers (shared by §4.2 figure + §4.4 figure/stats) -----

@dataclass(frozen=True)
class ModelUplift:
    """Per-model BTD ratings under both modes; both come from the same full-pool BTD."""

    domain_agent: dict[str, float]
    domain_baseline: dict[str, float]
    profile: str

    @property
    def agent_avg(self) -> float:
        return sum(self.domain_agent.values()) / len(self.domain_agent)

    @property
    def baseline_avg(self) -> float:
        return sum(self.domain_baseline.values()) / len(self.domain_baseline)

    @property
    def delta(self) -> float:
        return self.agent_avg - self.baseline_avg

    @property
    def per_domain_delta(self) -> dict[str, float]:
        """domain → Δ rating; only defined where both modes have data."""
        return {
            d: self.domain_agent[d] - self.domain_baseline[d]
            for d in self.domain_baseline
            if d in self.domain_agent
        }


def assign_tier(baseline_avg: float) -> str:
    """Map a baseline rating to one of TIER_THRESHOLDS keys."""
    for tier, (lo, hi) in TIER_THRESHOLDS.items():
        if lo <= baseline_avg < hi:
            return tier
    return "mid"


def collect_uplifts(
    *,
    judge: str = PRIMARY_JUDGE,
    config: str = CONSTRUCTION_PROFILE,
    whitelist: set[str] | None = None,
    exclude: set[str] | None = None,
    domains: tuple[str, ...] = ORDERED_DOMAINS,
) -> list[ModelUplift]:
    """Read full-pool arena leaderboards, build one ModelUplift per profile present in
    BOTH baseline:X and agent:X for at least one domain.

    Sorted by baseline_avg descending. Uses ``pool='full'`` so baseline:X and agent:X
    live on the same BTD scale.
    """
    domain_rows = {
        d: load_arena_leaderboard(d, judge=judge, config=config, pool="full")
        for d in domains
    }
    baseline_by_d = {d: rows_by_profile(domain_rows[d], mode="baseline") for d in domains}
    agent_by_d = {d: rows_by_profile(domain_rows[d], mode="agent") for d in domains}

    candidates: set[str] = set()
    for d in domains:
        candidates |= set(baseline_by_d[d]) & set(agent_by_d[d])
    if whitelist:
        candidates &= whitelist
    if exclude:
        candidates -= exclude

    out: list[ModelUplift] = []
    for profile in sorted(candidates):
        b_per: dict[str, float] = {}
        a_per: dict[str, float] = {}
        for d in domains:
            b = baseline_by_d[d].get(profile)
            a = agent_by_d[d].get(profile)
            if b is not None and a is not None:
                b_per[d] = b.rating
                a_per[d] = a.rating
        if not b_per:
            continue
        out.append(ModelUplift(
            domain_agent=a_per, domain_baseline=b_per, profile=profile,
        ))
    out.sort(key=lambda u: -u.baseline_avg)
    return out


# ---- output sinks --------------------------------------------------------

PAPER_TABLES_DIR = Path(__file__).resolve().parents[1] / "paper" / "tables"
PAPER_IMAGES_DIR = Path(__file__).resolve().parents[1] / "paper" / "images"


def write_text_artifact(target: Path, text: str) -> None:
    """Write paper-side artifact (.tex / .pdf) with a debug print to stdout."""
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"wrote {target}  ({len(text):,} bytes)")


def patch_table_body(target: Path, body_text: str) -> None:
    """Replace the body region of an existing LaTeX table file in place.

    Body region = everything between the FIRST `\\midrule` (which terminates the
    column-header row) and the next `\\bottomrule`. Caption, label, formatting,
    column headers, and any text outside the table environment are preserved.
    """
    if not target.exists():
        raise FileNotFoundError(
            f"--update-rows-only requires existing {target}; "
            f"run without the flag once to create it."
        )
    text = target.read_text(encoding="utf-8")
    midrule_idx = text.find("\\midrule")
    bottomrule_idx = text.find("\\bottomrule", midrule_idx) if midrule_idx != -1 else -1
    if midrule_idx == -1 or bottomrule_idx == -1:
        raise ValueError(
            f"Cannot find \\midrule…\\bottomrule structure in {target}; "
            f"file may not be a generated table."
        )
    body_start = text.find("\n", midrule_idx) + 1
    body = body_text.rstrip("\n") + "\n"
    new_text = text[:body_start] + body + text[bottomrule_idx:]
    target.write_text(new_text, encoding="utf-8")
    print(f"patched body of {target}  ({len(body):,} bytes; caption preserved)")
