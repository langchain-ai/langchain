"""Tests for the profile change summary generator."""

import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from langchain_model_profiles import cli
from langchain_model_profiles._summary import (
    _MAX_ROWS,
    FieldChange,
    ProfileDiff,
    ProfileParseError,
    _describe_new_model,
    _format_value,
    _truncate,
    build_summary,
    diff_profiles,
    extract_profiles,
    render_provider_section,
    summarize,
)

if TYPE_CHECKING:
    from langchain_core.language_models.model_profile import (
        ModelProfile,
        ModelProfileRegistry,
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
    """Absent `_PROFILES` yields `{}`; present-but-unparseable sources raise."""
    # No `_PROFILES` assignment, and an empty file, are both legitimately empty.
    assert extract_profiles("x = 1") == {}
    assert extract_profiles("") == {}
    # A syntactically broken file is corrupt, not empty.
    with pytest.raises(ProfileParseError):
        extract_profiles("def (:")
    # A non-literal or non-dict `_PROFILES` is corrupt too.
    with pytest.raises(ProfileParseError):
        extract_profiles("_PROFILES = some_function()")
    with pytest.raises(ProfileParseError):
        extract_profiles("_PROFILES = [1, 2, 3]")


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
    assert "across 1 provider(s)." in summary


def test_build_summary_no_changes() -> None:
    """An all-empty diff produces a short no-change note."""
    assert build_summary({"openai": ProfileDiff()}) == "No model profile data changed."


def test_truncation() -> None:
    """Long lists are truncated with a trailing count of hidden rows."""
    new: ModelProfileRegistry = {f"model-{i}": {"name": f"m{i}"} for i in range(40)}
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
    # The changed-field path is exercised end-to-end, not just at the unit layer.
    assert "max output tokens 4,096 → 16,384" in summary


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
    # Access the named fields before the tuple comparison: `== (1, 2)` would
    # otherwise narrow `change` to a plain `tuple` for the rest of the scope.
    assert change.old == 1
    assert change.new == 2
    assert change == (1, 2)


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
    old: ModelProfileRegistry = {"m": {"status": "active", "name": "M"}}
    new: ModelProfileRegistry = {"m": {"status": "deprecated", "name": "M2"}}
    section = render_provider_section("openai", diff_profiles(old, new))
    assert section is not None
    assert "status `active` → `deprecated`" in section
    assert "display name `M` → `M2`" in section


def test_render_removed_bool_field_change() -> None:
    """A boolean field flipped off renders a `removed <label>` phrase."""
    old: ModelProfileRegistry = {"m": {"image_inputs": True}}
    new: ModelProfileRegistry = {"m": {"image_inputs": False}}
    section = render_provider_section("openai", diff_profiles(old, new))
    assert section is not None
    assert "removed image input" in section


def test_describe_new_model_modalities() -> None:
    """A new model descriptor lists context, output, modalities, and tools."""
    profile: ModelProfile = {
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
    assert summary.index("<summary>aaa</summary>") < summary.index(
        "<summary>zzz</summary>"
    )


def test_build_summary_multi_provider_wraps_in_toggles() -> None:
    """More than one changed provider gets each wrapped in a <details> toggle."""
    diff_a = diff_profiles({}, {"a": {"name": "A"}})
    diff_z = diff_profiles({}, {"z": {"name": "Z"}})
    summary = build_summary({"aaa": diff_a, "zzz": diff_z})
    assert summary.count("<details>") == 2
    assert summary.count("</details>") == 2
    assert "<summary>aaa</summary>" in summary
    assert "<summary>zzz</summary>" in summary
    # The "### provider" headings are stripped inside toggles.
    assert "### aaa" not in summary
    assert "### zzz" not in summary
    # Headline counts only the wrapped providers.
    assert "across 2 provider(s)." in summary
    # Each toggle keeps its section body (guards against over-stripping): the
    # per-provider "N added" marker and model rows survive, and each row lands
    # after its own <summary> label rather than being dropped or misattributed.
    assert summary.count("1 added") == 2
    assert (
        summary.index("<summary>aaa</summary>")
        < summary.index("- `a`")
        < summary.index("<summary>zzz</summary>")
        < summary.index("- `z`")
    )


def test_build_summary_multi_provider_preserves_removed_and_changed() -> None:
    """Multi-provider toggles keep removed and changed bodies, not just added."""
    diff = diff_profiles(extract_profiles(_OLD_SOURCE), extract_profiles(_NEW_SOURCE))
    other = diff_profiles({}, {"m": {"name": "M"}})
    summary = build_summary({"openai": diff, "zzz": other})
    assert summary.count("<details>") == 2
    # The realistic diff's removed and changed phrases survive the heading strip.
    assert "1 removed" in summary
    assert "`old-model`" in summary
    assert "max output tokens 4,096 → 16,384" in summary
    # Changed content stays inside the openai toggle, before the next provider.
    assert (
        summary.index("<summary>openai</summary>")
        < summary.index("max output tokens 4,096 → 16,384")
        < summary.index("<summary>zzz</summary>")
    )


def test_build_summary_single_provider_no_toggle() -> None:
    """A single changed provider renders as a plain section without toggles."""
    diff = diff_profiles(extract_profiles(_OLD_SOURCE), extract_profiles(_NEW_SOURCE))
    summary = build_summary({"openai": diff})
    assert "<details>" not in summary
    assert "### openai" in summary
    assert "across 1 provider(s)." in summary


def test_build_summary_empty_diff_filtered_from_count() -> None:
    """Providers with empty diffs are excluded from the toggle/count decision."""
    diff_a = diff_profiles({}, {"a": {"name": "A"}})
    summary = build_summary({"aaa": diff_a, "empty": ProfileDiff()})
    # Only one provider actually changed, so no toggles and the count is 1.
    assert "<details>" not in summary
    assert "### aaa" in summary
    assert "across 1 provider(s)." in summary


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
        summarize([{"provider": "openai"}], repo_root=repo)  # type: ignore[typeddict-item]


def test_summarize_non_string_entry(tmp_path: Path) -> None:
    """Non-string `provider`/`data_dir` raises `TypeError`, not `AttributeError`.

    A non-string value would otherwise reach `data_dir.rstrip(...)` and raise an
    `AttributeError` that escapes the CLI's `except (RuntimeError, ValueError,
    TypeError)`, surfacing a raw traceback instead of a clean error.
    """
    repo = tmp_path
    _init_repo(repo)
    (repo / "README.md").write_text("x")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    with pytest.raises(TypeError, match="must be strings"):
        summarize(
            [{"provider": 5, "data_dir": 7}],  # type: ignore[typeddict-item]
            repo_root=repo,
        )


def test_summarize_corrupt_working_tree_file(tmp_path: Path) -> None:
    """A present-but-unparseable working-tree file raises, not a mass removal.

    Mirrors the `_verify_ref` guard on the base-ref side: a corrupt new file
    must surface as an error rather than be diffed as every model removed.
    """
    repo = tmp_path
    _init_repo(repo)
    data_dir = "libs/partners/openai/data"
    profiles_path = repo / data_dir / "_profiles.py"
    _write_profiles(profiles_path, _OLD_SOURCE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    # Simulate a refresh that left the file truncated / syntactically broken.
    profiles_path.write_text("_PROFILES = {")

    with pytest.raises(RuntimeError, match="unparseable"):
        summarize([{"provider": "openai", "data_dir": data_dir}], repo_root=repo)


def test_summarize_corrupt_base_ref_file(tmp_path: Path) -> None:
    """An unparseable file at the base ref raises, not an all-added diff."""
    repo = tmp_path
    _init_repo(repo)
    data_dir = "libs/partners/openai/data"
    profiles_path = repo / data_dir / "_profiles.py"
    _write_profiles(profiles_path, "_PROFILES = {")  # committed broken
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")

    profiles_path.write_text(_NEW_SOURCE)  # working tree now valid

    with pytest.raises(RuntimeError, match="unparseable"):
        summarize([{"provider": "openai", "data_dir": data_dir}], repo_root=repo)


def test_cli_summarize_success(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A valid `--providers` array prints the Markdown summary to stdout."""
    repo = tmp_path
    _init_repo(repo)
    data_dir = "libs/partners/openai/data"
    profiles_path = repo / data_dir / "_profiles.py"
    _write_profiles(profiles_path, _OLD_SOURCE)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "init")
    profiles_path.write_text(_NEW_SOURCE)

    providers = json.dumps([{"provider": "openai", "data_dir": data_dir}])
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "langchain-profiles",
            "summarize",
            "--providers",
            providers,
            "--repo-root",
            str(repo),
        ],
    )

    cli.main()

    out = capsys.readouterr().out
    assert "## Summary of changes" in out
    assert "`gpt-5`" in out
    assert "max output tokens 4,096 → 16,384" in out


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
