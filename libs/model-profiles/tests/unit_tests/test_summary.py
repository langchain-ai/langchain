"""Tests for the profile change summary generator."""

import subprocess
from pathlib import Path

from langchain_model_profiles.summary import (
    ProfileDiff,
    build_summary,
    diff_profiles,
    extract_profiles,
    render_provider_section,
    summarize,
)

_OLD_SOURCE = '''"""Auto-generated."""

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    "gpt-4": {
        "name": "GPT-4",
        "max_input_tokens": 8192,
        "max_output_tokens": 4096,
        "image_inputs": False,
        "tool_calling": True,
    },
    "old-model": {
        "name": "Old",
        "max_input_tokens": 1000,
    },
}
'''

_NEW_SOURCE = '''"""Auto-generated."""

from typing import Any

_PROFILES: dict[str, dict[str, Any]] = {
    "gpt-4": {
        "name": "GPT-4",
        "max_input_tokens": 8192,
        "max_output_tokens": 16384,
        "image_inputs": True,
        "tool_calling": True,
    },
    "gpt-5": {
        "name": "GPT-5",
        "max_input_tokens": 400000,
        "max_output_tokens": 128000,
        "image_inputs": True,
        "reasoning_output": True,
        "tool_calling": True,
    },
}
'''


def _git(repo: Path, *args: str) -> None:
    """Run a git command inside `repo` (test helper)."""
    subprocess.run(  # noqa: S603
        ["git", "-C", str(repo), *args],  # noqa: S607
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(repo: Path) -> None:
    """Initialize a git repo with a deterministic identity."""
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "Test")


def _write_profiles(path: Path, source: str) -> None:
    """Write a `_profiles.py` file, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source)


def test_extract_profiles() -> None:
    """`_PROFILES` literal is extracted via ast without executing the module."""
    profiles = extract_profiles(_OLD_SOURCE)
    assert set(profiles) == {"gpt-4", "old-model"}
    assert profiles["gpt-4"]["max_input_tokens"] == 8192


def test_extract_profiles_handles_missing_or_invalid() -> None:
    """Missing or unparseable sources yield an empty mapping."""
    assert extract_profiles("x = 1") == {}
    assert extract_profiles("def (:") == {}


def test_diff_profiles() -> None:
    """Diff reports added, removed, and per-field changes."""
    diff = diff_profiles(extract_profiles(_OLD_SOURCE), extract_profiles(_NEW_SOURCE))
    assert diff.added == ["gpt-5"]
    assert diff.removed == ["old-model"]
    assert set(diff.changed) == {"gpt-4"}
    assert diff.changed["gpt-4"]["max_output_tokens"] == (4096, 16384)
    assert diff.changed["gpt-4"]["image_inputs"] == (False, True)
    assert diff.added_profiles["gpt-5"]["max_input_tokens"] == 400000


def test_diff_profiles_no_changes() -> None:
    """Identical inputs produce an empty diff."""
    profiles = extract_profiles(_OLD_SOURCE)
    diff = diff_profiles(profiles, profiles)
    assert diff.is_empty


def test_render_provider_section_content() -> None:
    """Rendered section describes additions, removals, and field changes."""
    diff = diff_profiles(extract_profiles(_OLD_SOURCE), extract_profiles(_NEW_SOURCE))
    section = render_provider_section("openai", diff)
    assert section is not None
    assert "### openai" in section
    assert "1 added" in section
    assert "`gpt-5`" in section
    assert "400,000 ctx" in section
    assert "reasoning" in section
    assert "1 removed" in section
    assert "`old-model`" in section
    assert "1 changed" in section
    assert "max output tokens 4,096 → 16,384" in section
    assert "added image input" in section


def test_render_provider_section_empty() -> None:
    """An empty diff renders no section."""
    assert render_provider_section("openai", ProfileDiff()) is None


def test_build_summary_headline() -> None:
    """The summary leads with a header and an aggregate headline."""
    diff = diff_profiles(extract_profiles(_OLD_SOURCE), extract_profiles(_NEW_SOURCE))
    summary = build_summary({"openai": diff})
    assert summary.startswith("## Summary of changes")
    assert "1 added" in summary
    assert "1 removed" in summary
    assert "1 changed" in summary


def test_build_summary_no_changes() -> None:
    """An all-empty diff produces a short no-change note."""
    assert build_summary({"openai": ProfileDiff()}) == "No model profile data changed."


def test_truncation() -> None:
    """Long lists are truncated with a trailing count of hidden rows."""
    new = {f"model-{i}": {"name": f"m{i}"} for i in range(40)}
    diff = diff_profiles({}, new)
    section = render_provider_section("openai", diff)
    assert section is not None
    assert "…and 15 more" in section


def test_summarize_against_git(tmp_path: Path) -> None:
    """Summarize compares the working tree against a committed baseline."""
    repo = tmp_path
    _init_repo(repo)

    data_dir = "libs/partners/openai/data"
    profiles_path = repo / data_dir / "_profiles.py"
    _write_profiles(profiles_path, _OLD_SOURCE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    # Simulate a refresh by overwriting the working-tree file.
    profiles_path.write_text(_NEW_SOURCE)

    summary = summarize([{"provider": "openai", "data_dir": data_dir}], repo_root=repo)
    assert "## Summary of changes" in summary
    assert "`gpt-5`" in summary
    assert "`old-model`" in summary


def test_summarize_new_provider_file(tmp_path: Path) -> None:
    """A brand-new profiles file is treated as all-added."""
    repo = tmp_path
    _init_repo(repo)
    (repo / "README.md").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    data_dir = "libs/partners/new/data"
    _write_profiles(repo / data_dir / "_profiles.py", _NEW_SOURCE)

    summary = summarize([{"provider": "new", "data_dir": data_dir}], repo_root=repo)
    assert "2 added" in summary
