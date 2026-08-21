"""Validation for `SKILL.md` frontmatter and skill directory structure.

A malformed `SKILL.md` is often skipped silently or fails at agent runtime with
little indication of what went wrong. `validate_skill_file` and
`validate_skill_directory` catch these issues ahead of time (e.g. in CI) by
checking that a skill's YAML frontmatter is well-formed and complete, and that
a directory of skill packs follows the expected `<skill-name>/SKILL.md` layout.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

if TYPE_CHECKING:
    from os import PathLike

# Anthropic's skill naming convention: lowercase letters, digits, and single hyphens.
_NAME_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
_MAX_NAME_LENGTH = 64
_MAX_DESCRIPTION_LENGTH = 1024
_FRONTMATTER_DELIMITER = "---"


@dataclass
class SkillValidationReport:
    """Result of validating a single `SKILL.md` file.

    Attributes:
        path: Path to the validated `SKILL.md` file (or skill directory, if the
            file itself could not be found).
        errors: Problems that make the skill invalid.
        warnings: Problems that don't invalidate the skill but should be
            addressed (e.g. an overly long description).
    """

    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Whether the skill has no validation errors.

        A report can be valid while still having warnings.
        """
        return not self.errors


def _split_frontmatter(text: str) -> tuple[str, str] | None:
    """Split `SKILL.md` content into raw YAML frontmatter and body text.

    Args:
        text: Raw content of the `SKILL.md` file.

    Returns:
        A tuple of `(raw_frontmatter, body)`, or `None` if the file doesn't
        start with a `---` delimited frontmatter block.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIMITER:
        return None

    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONTMATTER_DELIMITER:
            raw_frontmatter = "\n".join(lines[1:index])
            body = "\n".join(lines[index + 1 :])
            return raw_frontmatter, body

    return None


def _parse_frontmatter(text: str) -> dict[str, Any]:
    """Parse the YAML frontmatter out of `SKILL.md` content.

    Args:
        text: Raw content of the `SKILL.md` file.

    Returns:
        The parsed frontmatter mapping.

    Raises:
        ValueError: If the frontmatter block is missing or isn't valid YAML.
        TypeError: If the frontmatter doesn't parse to a mapping.
    """
    split = _split_frontmatter(text)
    if split is None:
        msg = "Missing YAML frontmatter: file must start with a '---' delimited block."
        raise ValueError(msg)

    raw_frontmatter, _body = split
    try:
        frontmatter = yaml.safe_load(raw_frontmatter)
    except yaml.YAMLError as e:
        msg = f"Invalid YAML frontmatter: {e}"
        raise ValueError(msg) from e

    if not isinstance(frontmatter, dict):
        msg = "Frontmatter must be a YAML mapping of key-value pairs."
        raise TypeError(msg)

    return frontmatter


def validate_skill_file(path: str | PathLike[str]) -> SkillValidationReport:
    """Validate a single `SKILL.md` file's frontmatter and contents.

    Checks that the file exists, has valid YAML frontmatter delimited by `---`
    markers, and that the frontmatter declares a well-formed `name` (lowercase
    alphanumeric characters and hyphens, at most 64 characters) and a
    non-empty `description` (warning, not erroring, past 1024 characters).

    Args:
        path: Path to the `SKILL.md` file to validate.

    Returns:
        A `SkillValidationReport` describing any errors or warnings found.
    """
    path = Path(path)
    report = SkillValidationReport(path=path)

    if not path.is_file():
        report.errors.append(f"File not found: {path}")
        return report

    text = path.read_text(encoding="utf-8")
    try:
        frontmatter = _parse_frontmatter(text)
    except (ValueError, TypeError) as e:
        report.errors.append(str(e))
        return report

    name = frontmatter.get("name")
    if not name:
        report.errors.append("Frontmatter is missing required field 'name'.")
    elif not isinstance(name, str):
        report.errors.append("Frontmatter field 'name' must be a string.")
    else:
        if len(name) > _MAX_NAME_LENGTH:
            report.errors.append(
                f"Frontmatter field 'name' must be at most {_MAX_NAME_LENGTH} "
                f"characters (got {len(name)})."
            )
        if not _NAME_PATTERN.match(name):
            report.errors.append(
                "Frontmatter field 'name' must contain only lowercase "
                "alphanumeric characters separated by single hyphens "
                "(e.g. 'code-review')."
            )

    description = frontmatter.get("description")
    if not description:
        report.errors.append("Frontmatter is missing required field 'description'.")
    elif not isinstance(description, str):
        report.errors.append("Frontmatter field 'description' must be a string.")
    elif len(description) > _MAX_DESCRIPTION_LENGTH:
        report.warnings.append(
            f"Frontmatter field 'description' exceeds the recommended "
            f"{_MAX_DESCRIPTION_LENGTH} character limit (got {len(description)})."
        )

    return report


def validate_skill_directory(path: str | PathLike[str]) -> list[SkillValidationReport]:
    """Validate every skill in a directory of skill packs.

    Each immediate subdirectory of `path` is expected to contain a `SKILL.md`
    file, following the `<skill-name>/SKILL.md` layout convention (e.g.
    `skills/code-review/SKILL.md`). A subdirectory whose frontmatter `name`
    doesn't match its directory name produces a warning, since agents that
    key off the directory name would silently load the wrong skill identity.

    Args:
        path: Path to the directory containing skill subdirectories.

    Returns:
        A list of `SkillValidationReport`, one per skill subdirectory found.
        A missing directory or one with no subdirectories produces a single
        report with an error and no other checks performed.
    """
    root = Path(path)

    if not root.is_dir():
        return [SkillValidationReport(path=root, errors=[f"Directory not found: {root}"])]

    skill_dirs = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name)
    if not skill_dirs:
        return [SkillValidationReport(path=root, errors=[f"No skill directories found in: {root}"])]

    reports = []
    for skill_dir in skill_dirs:
        skill_file = skill_dir / "SKILL.md"
        if not skill_file.is_file():
            reports.append(
                SkillValidationReport(
                    path=skill_dir,
                    errors=[f"Missing SKILL.md file in skill directory: {skill_dir}"],
                )
            )
            continue

        report = validate_skill_file(skill_file)

        try:
            frontmatter = _parse_frontmatter(skill_file.read_text(encoding="utf-8"))
        except (ValueError, TypeError):
            pass
        else:
            name = frontmatter.get("name")
            if isinstance(name, str) and name != skill_dir.name:
                report.warnings.append(
                    f"Frontmatter name '{name}' does not match its directory "
                    f"name '{skill_dir.name}'."
                )

        reports.append(report)

    return reports
