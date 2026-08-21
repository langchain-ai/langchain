"""Unit tests for `langchain.skills.validator`."""

from pathlib import Path

import pytest

from langchain.skills.validator import (
    SkillValidationReport,
    validate_skill_directory,
    validate_skill_file,
)


def _write_skill(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_valid_skill_file(tmp_path: Path) -> None:
    skill_file = _write_skill(
        tmp_path / "SKILL.md",
        "---\nname: code-review\ndescription: Reviews pull requests for bugs.\n---\nBody text.\n",
    )

    report = validate_skill_file(skill_file)

    assert report.is_valid
    assert report.errors == []
    assert report.warnings == []


def test_missing_file() -> None:
    report = validate_skill_file("/nonexistent/SKILL.md")

    assert not report.is_valid
    assert len(report.errors) == 1
    assert "File not found" in report.errors[0]


def test_missing_frontmatter_delimiter(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path / "SKILL.md", "name: code-review\ndescription: A skill.\n")

    report = validate_skill_file(skill_file)

    assert not report.is_valid
    assert any("frontmatter" in error.lower() for error in report.errors)


def test_invalid_yaml_syntax(tmp_path: Path) -> None:
    skill_file = _write_skill(
        tmp_path / "SKILL.md",
        "---\nname: code-review\n  bad_indent: [unterminated\n---\nBody.\n",
    )

    report = validate_skill_file(skill_file)

    assert not report.is_valid
    assert any("Invalid YAML" in error for error in report.errors)


def test_frontmatter_not_a_mapping(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path / "SKILL.md", "---\n- just\n- a\n- list\n---\nBody.\n")

    report = validate_skill_file(skill_file)

    assert not report.is_valid
    assert any("mapping" in error.lower() for error in report.errors)


def test_missing_required_fields(tmp_path: Path) -> None:
    skill_file = _write_skill(tmp_path / "SKILL.md", "---\nfoo: bar\n---\nBody.\n")

    report = validate_skill_file(skill_file)

    assert not report.is_valid
    assert any("'name'" in error for error in report.errors)
    assert any("'description'" in error for error in report.errors)


@pytest.mark.parametrize(
    "name",
    ["Code-Review", "code_review", "code review", "-code-review", "code-review-", "code--review"],
)
def test_invalid_name_format(tmp_path: Path, name: str) -> None:
    skill_file = _write_skill(
        tmp_path / "SKILL.md",
        f"---\nname: {name}\ndescription: A skill.\n---\nBody.\n",
    )

    report = validate_skill_file(skill_file)

    assert not report.is_valid
    assert any("lowercase" in error.lower() for error in report.errors)


def test_name_too_long(tmp_path: Path) -> None:
    long_name = "a" * 65
    skill_file = _write_skill(
        tmp_path / "SKILL.md",
        f"---\nname: {long_name}\ndescription: A skill.\n---\nBody.\n",
    )

    report = validate_skill_file(skill_file)

    assert not report.is_valid
    assert any("at most 64" in error for error in report.errors)


def test_description_too_long_is_a_warning_not_an_error(tmp_path: Path) -> None:
    long_description = "a" * 1025
    skill_file = _write_skill(
        tmp_path / "SKILL.md",
        f"---\nname: code-review\ndescription: {long_description}\n---\nBody.\n",
    )

    report = validate_skill_file(skill_file)

    assert report.is_valid
    assert len(report.warnings) == 1
    assert "1024" in report.warnings[0]


def test_validate_skill_directory_happy_path(tmp_path: Path) -> None:
    skill_dir = tmp_path / "code-review"
    skill_dir.mkdir()
    _write_skill(
        skill_dir / "SKILL.md",
        "---\nname: code-review\ndescription: Reviews pull requests.\n---\nBody.\n",
    )

    reports = validate_skill_directory(tmp_path)

    assert len(reports) == 1
    assert reports[0].is_valid
    assert reports[0].path == skill_dir / "SKILL.md"


def test_validate_skill_directory_missing_skill_md(tmp_path: Path) -> None:
    skill_dir = tmp_path / "code-review"
    skill_dir.mkdir()

    reports = validate_skill_directory(tmp_path)

    assert len(reports) == 1
    assert not reports[0].is_valid
    assert "Missing SKILL.md" in reports[0].errors[0]


def test_validate_skill_directory_name_mismatch_warns(tmp_path: Path) -> None:
    skill_dir = tmp_path / "code-review"
    skill_dir.mkdir()
    _write_skill(
        skill_dir / "SKILL.md",
        "---\nname: something-else\ndescription: Reviews pull requests.\n---\nBody.\n",
    )

    reports = validate_skill_directory(tmp_path)

    assert len(reports) == 1
    assert reports[0].is_valid
    assert any("does not match" in warning for warning in reports[0].warnings)


def test_validate_skill_directory_missing_root(tmp_path: Path) -> None:
    reports = validate_skill_directory(tmp_path / "does-not-exist")

    assert len(reports) == 1
    assert not reports[0].is_valid
    assert "Directory not found" in reports[0].errors[0]


def test_validate_skill_directory_empty(tmp_path: Path) -> None:
    reports = validate_skill_directory(tmp_path)

    assert len(reports) == 1
    assert not reports[0].is_valid
    assert "No skill directories found" in reports[0].errors[0]


def test_validate_skill_directory_multiple_skills(tmp_path: Path) -> None:
    good_dir = tmp_path / "code-review"
    good_dir.mkdir()
    _write_skill(
        good_dir / "SKILL.md",
        "---\nname: code-review\ndescription: Reviews pull requests.\n---\nBody.\n",
    )

    bad_dir = tmp_path / "broken-skill"
    bad_dir.mkdir()
    _write_skill(bad_dir / "SKILL.md", "no frontmatter here\n")

    reports = validate_skill_directory(tmp_path)

    assert len(reports) == 2
    reports_by_name = {report.path.parent.name: report for report in reports}
    assert reports_by_name["code-review"].is_valid
    assert not reports_by_name["broken-skill"].is_valid


def test_skill_validation_report_is_valid_with_only_warnings() -> None:
    report = SkillValidationReport(path=Path("SKILL.md"), warnings=["a warning"])

    assert report.is_valid
