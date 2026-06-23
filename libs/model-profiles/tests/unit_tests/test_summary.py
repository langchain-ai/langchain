"""Tests for the profile change summary generator."""

import subprocess
import sys
from pathlib import Path

import pytest

from langchain_model_profiles import cli
from langchain_model_profiles._summary import (
    _MAX_ROWS,
    FieldChange,
    ProfileDiff,
    _describe_new_model,
    _format_value,
    _truncate,
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


def test_field_change_is_tuple() -> None:
    """`FieldChange` unpacks and compares like a plain (old, new) tuple."""
    change = FieldChange(1, 2)
    assert change == (1, 2)
    assert change.old == 1
    assert change.new == 2


def test_format_value_variants() -> None:
    """Each `_format_value` branch renders the expected string."""
    assert _format_value("x", None) == "unset"
    assert _format_value("tool_calling", True) == "yes"  # noqa: FBT003
    assert _format_value("tool_calling", False) == "no"  # noqa: FBT003
    assert _format_value("max_input_tokens", 200000) == "200,000"
    # Plain int outside the token fields is rendered without separators.
    assert _format_value("foo", 42) == "42"
    # Floats fall through to str().
    assert _format_value("temperature", 1.5) == "1.5"
    assert _format_value("name", "GPT") == "`GPT`"


def test_render_non_bool_field_change() -> None:
    """Non-boolean field changes render an `old → new` phrase."""
    old = {"m": {"status": "active", "name": "M"}}
    new = {"m": {"status": "deprecated", "name": "M2"}}
    section = render_provider_section("openai", diff_profiles(old, new))
    assert section is not None
    assert "status `active` → `deprecated`" in section
    assert "display name `M` → `M2`" in section


def test_describe_new_model_modalities() -> None:
    """A new model descriptor lists context, output, modalities, and tools."""
    profile = {
        "max_input_tokens": 200000,
        "max_output_tokens": 64000,
        "image_inputs": True,
        "audio_inputs": True,
        "video_inputs": True,
        "pdf_inputs": True,
        "tool_calling": True,
    }
    descriptor = _describe_new_model(profile)
    assert "200,000 ctx" in descriptor
    assert "64,000 out" in descriptor
    assert "text+image+audio+video+pdf in" in descriptor
    assert "tools" in descriptor


def test_describe_new_model_empty() -> None:
    """A profile with no notable fields yields an empty descriptor."""
    assert _describe_new_model({"name": "x"}) == ""


def test_render_added_model_without_descriptor() -> None:
    """An added model with no descriptor renders no ` — ` suffix."""
    section = render_provider_section("p", diff_profiles({}, {"bare": {"name": "B"}}))
    assert section is not None
    assert "- `bare`" in section
    assert "- `bare` —" not in section


def test_truncate_boundary() -> None:
    """`_truncate` keeps exactly `_MAX_ROWS` rows but caps one more."""
    exactly = [f"- r{i}" for i in range(_MAX_ROWS)]
    assert _truncate(exactly) == exactly

    over = [f"- r{i}" for i in range(_MAX_ROWS + 1)]
    result = _truncate(over)
    assert len(result) == _MAX_ROWS + 1
    assert result[-1] == "- …and 1 more"


def test_build_summary_multi_provider_sorted() -> None:
    """Providers are rendered in sorted order regardless of input order."""
    diff_a = diff_profiles({}, {"a": {"name": "A"}})
    diff_z = diff_profiles({}, {"z": {"name": "Z"}})
    summary = build_summary({"zzz": diff_z, "aaa": diff_a})
    assert summary.index("### aaa") < summary.index("### zzz")


def test_summarize_removed_when_file_deleted(tmp_path: Path) -> None:
    """Deleting the working-tree file reports every model as removed."""
    repo = tmp_path
    _init_repo(repo)
    data_dir = "libs/partners/openai/data"
    profiles_path = repo / data_dir / "_profiles.py"
    _write_profiles(profiles_path, _OLD_SOURCE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    profiles_path.unlink()

    summary = summarize([{"provider": "openai", "data_dir": data_dir}], repo_root=repo)
    assert "2 removed" in summary


def test_summarize_bad_base_ref(tmp_path: Path) -> None:
    """An unresolvable base ref raises rather than fabricating an all-added diff."""
    repo = tmp_path
    _init_repo(repo)
    (repo / "README.md").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    data_dir = "libs/partners/openai/data"
    _write_profiles(repo / data_dir / "_profiles.py", _NEW_SOURCE)

    with pytest.raises(RuntimeError, match="Could not resolve base ref"):
        summarize(
            [{"provider": "openai", "data_dir": data_dir}],
            base_ref="no-such-ref",
            repo_root=repo,
        )


def test_summarize_malformed_entry(tmp_path: Path) -> None:
    """A provider entry missing a required key raises a clear error."""
    repo = tmp_path
    _init_repo(repo)
    (repo / "README.md").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    with pytest.raises(ValueError, match="Invalid provider entry"):
        summarize([{"provider": "openai"}], repo_root=repo)  # type: ignore[list-item]


def test_cli_summarize_invalid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI rejects a `--providers` value that is not valid JSON."""
    monkeypatch.setattr(
        sys, "argv", ["langchain-profiles", "summarize", "--providers", "not json"]
    )
    with pytest.raises(SystemExit):
        cli.main()


def test_cli_summarize_non_list(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI rejects a `--providers` value that is not a JSON array."""
    monkeypatch.setattr(
        sys, "argv", ["langchain-profiles", "summarize", "--providers", '{"a": 1}']
    )
    with pytest.raises(SystemExit):
        cli.main()
