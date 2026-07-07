"""Loader for analytical skill libraries (SKILL.md files under skills/)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_SKILLS_DIR = Path(__file__).parent / "skills"
_FRONTMATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_DESCRIPTION_RE = re.compile(r"^description:\s*(>|\|)?\s*(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class Skill:
    """One analytical skill: name, one-line description, full body for prompt injection."""

    body: str
    description: str
    name: str


def _parse_description(frontmatter: str) -> str:
    """Extract the YAML 'description' field, supporting folded scalars."""
    match = _DESCRIPTION_RE.search(frontmatter)
    if not match:
        return ""
    folded, first_line = match.group(1), match.group(2).strip()
    if folded not in (">", "|"):
        return first_line.strip("'\"")
    parts: list[str] = []
    after = frontmatter[match.end():]
    for line in after.splitlines():
        if not line.strip():
            if parts:
                break
            continue
        if line.startswith((" ", "\t")):
            parts.append(line.strip())
        else:
            break
    return " ".join(parts)


def _parse_skill(path: Path) -> Skill:
    text = path.read_text(encoding="utf-8")
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise ValueError(f"Skill {path} is missing YAML frontmatter")
    description = _parse_description(match.group(1))
    body = text[match.end():].strip()
    return Skill(body=body, description=description, name=path.parent.name)


def load_skills(skills_dir: Path = _SKILLS_DIR) -> dict[str, Skill]:
    """Load every SKILL.md under skills_dir into a {name: Skill} mapping."""
    out: dict[str, Skill] = {}
    for entry in sorted(skills_dir.iterdir()):
        skill_md = entry / "SKILL.md"
        if not skill_md.is_file():
            continue
        skill = _parse_skill(skill_md)
        out[skill.name] = skill
    if not out:
        raise RuntimeError(f"No skills found under {skills_dir}")
    return out


SKILLS: dict[str, Skill] = load_skills()
SKILL_NAMES: tuple[str, ...] = tuple(SKILLS)
