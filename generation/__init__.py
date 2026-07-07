"""Generation stage: Baseline (single-pass) and Agent (skill-pipeline) Submission writers."""

from .agents import (
    agent_final_writer,
    agent_intermediate_writer,
    base_writer_agent,
    skill_selector_agent,
)
from .runtime import (
    MAX_SKILLS_PER_CASE,
    encode_submission,
    generate_submission,
    run_agent,
    run_baseline,
    select_skills,
)
from .schema import (
    CategorizedHypothesisCandidate,
    HypothesisCandidate,
    HypothesisCandidateSet,
    SkillSelection,
)
from .skills import SKILL_NAMES, SKILLS, Skill

__all__ = [
    "CategorizedHypothesisCandidate",
    "HypothesisCandidate",
    "HypothesisCandidateSet",
    "MAX_SKILLS_PER_CASE",
    "SKILLS",
    "SKILL_NAMES",
    "Skill",
    "SkillSelection",
    "agent_final_writer",
    "agent_intermediate_writer",
    "base_writer_agent",
    "encode_submission",
    "generate_submission",
    "run_agent",
    "run_baseline",
    "select_skills",
    "skill_selector_agent",
]
