"""Generate a plain-English summary of model profile changes.

The `refresh_model_profiles` workflow opens an automated PR whenever the data
behind `_profiles.py` files changes. Those diffs are large blocks of generated
data, so a reviewer otherwise has to open *Files changed* and eyeball raw values
to learn what actually moved. This module turns the structured before/after data
into a skimmable Markdown summary (new models, removed models, and per-field
capability/metadata changes) for the PR body. The summary is generated
deterministically from the data, so there is no risk of an LLM misdescribing it.
"""

from __future__ import annotations

import ast
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

if TYPE_CHECKING:
    from collections.abc import Mapping

    from langchain_core.language_models.model_profile import (
        ModelProfile,
        ModelProfileRegistry,
    )

# Maximum number of bullet rows rendered per section before truncating.
_MAX_ROWS = 25

# Human-readable labels for profile fields.
_FIELD_LABELS: dict[str, str] = {
    "name": "display name",
    "status": "status",
    "release_date": "release date",
    "last_updated": "last updated",
    "open_weights": "open weights",
    "max_input_tokens": "max input tokens",
    "max_output_tokens": "max output tokens",
    "text_inputs": "text input",
    "image_inputs": "image input",
    "audio_inputs": "audio input",
    "pdf_inputs": "PDF input",
    "video_inputs": "video input",
    "text_outputs": "text output",
    "image_outputs": "image output",
    "audio_outputs": "audio output",
    "video_outputs": "video output",
    "reasoning_output": "reasoning",
    "tool_calling": "tool calling",
    "tool_choice": "tool choice",
    "tool_call_streaming": "tool call streaming",
    "structured_output": "structured output",
    "attachment": "attachments",
    "temperature": "temperature control",
    "image_url_inputs": "image URL input",
    "image_tool_message": "image tool messages",
    "pdf_tool_message": "PDF tool messages",
}

# Token fields rendered with thousands separators.
_TOKEN_FIELDS = frozenset({"max_input_tokens", "max_output_tokens"})


class FieldChange(NamedTuple):
    """Old and new values for a single changed profile field.

    Named rather than a bare `tuple` so the old → new ordering the renderer
    relies on is part of the type instead of a positional convention. Values are
    heterogeneous profile data (bool, int, str, or unset), so `Any` is the
    honest element type here.
    """

    old: Any
    """Value before the refresh, or `None` if the field was unset."""

    new: Any
    """Value after the refresh, or `None` if the field is now unset."""


class ProviderEntry(TypedDict):
    """One provider's identity and the data dir holding its `_profiles.py`."""

    provider: str
    """Provider identifier (e.g. `'openai'`)."""

    data_dir: str
    """Path to the provider's data directory, relative to the repo root."""


@dataclass
class ProfileDiff:
    """Structured difference between two sets of model profiles."""

    added: list[str] = field(default_factory=list)
    """Model IDs present after the refresh but not before, sorted."""

    removed: list[str] = field(default_factory=list)
    """Model IDs present before the refresh but not after, sorted."""

    changed: dict[str, dict[str, FieldChange]] = field(default_factory=dict)
    """Per-model field changes, keyed by model ID then field name."""

    added_profiles: ModelProfileRegistry = field(default_factory=dict)
    """Full profiles for each added model, keyed by model ID."""

    @property
    def is_empty(self) -> bool:
        """Whether there are no model additions, removals, or field changes."""
        return not (self.added or self.removed or self.changed)


def extract_profiles(source: str) -> ModelProfileRegistry:
    """Extract the `_PROFILES` mapping from `_profiles.py` source.

    Uses `ast.literal_eval` rather than importing/executing the module so the
    generated data file is never run as code.

    Args:
        source: Contents of a `_profiles.py` module.

    Returns:
        The `_PROFILES` mapping, or an empty dict if it cannot be found.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}

    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            targets: list[ast.expr] = [node.target]
        elif isinstance(node, ast.Assign):
            targets = list(node.targets)
        else:
            continue
        is_profiles = any(
            isinstance(t, ast.Name) and t.id == "_PROFILES" for t in targets
        )
        if is_profiles and node.value is not None:
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError):
                return {}
            return value if isinstance(value, dict) else {}
    return {}


def diff_profiles(old: ModelProfileRegistry, new: ModelProfileRegistry) -> ProfileDiff:
    """Compute the difference between two `_PROFILES` mappings.

    Args:
        old: Profiles before the refresh.
        new: Profiles after the refresh.

    Returns:
        A `ProfileDiff` describing added, removed, and changed models.
    """
    added = sorted(set(new) - set(old))
    removed = sorted(set(old) - set(new))

    changed: dict[str, dict[str, FieldChange]] = {}
    for model_id in sorted(set(old) & set(new)):
        # View profiles as plain mappings so we can iterate dynamic keys (the
        # `ModelProfile` TypedDict only permits literal-key access).
        old_profile: Mapping[str, Any] = old[model_id]
        new_profile: Mapping[str, Any] = new[model_id]
        fields: dict[str, FieldChange] = {}
        for key in sorted(set(old_profile) | set(new_profile)):
            old_val = old_profile.get(key)
            new_val = new_profile.get(key)
            if old_val != new_val:
                fields[key] = FieldChange(old_val, new_val)
        if fields:
            changed[model_id] = fields

    added_profiles = {model_id: new[model_id] for model_id in added}
    return ProfileDiff(
        added=added,
        removed=removed,
        changed=changed,
        added_profiles=added_profiles,
    )


def _format_value(field_name: str, value: Any) -> str:  # noqa: ANN401
    """Render a single field value for display."""
    if value is None:
        return "unset"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int) and field_name in _TOKEN_FIELDS:
        return f"{value:,}"
    return f"`{value}`" if isinstance(value, str) else str(value)


def _describe_field_change(
    field_name: str,
    old_val: Any,  # noqa: ANN401
    new_val: Any,  # noqa: ANN401
) -> str:
    """Produce a plain-English phrase for one field change."""
    label = _FIELD_LABELS.get(field_name, field_name)
    if isinstance(old_val, bool) or isinstance(new_val, bool):
        if new_val and not old_val:
            return f"added {label}"
        if old_val and not new_val:
            return f"removed {label}"
    old_str = _format_value(field_name, old_val)
    new_str = _format_value(field_name, new_val)
    return f"{label} {old_str} → {new_str}"


def _describe_new_model(profile: ModelProfile) -> str:
    """Produce a short descriptor for a newly added model."""
    parts: list[str] = []
    context = profile.get("max_input_tokens")
    if context:
        parts.append(f"{context:,} ctx")
    output = profile.get("max_output_tokens")
    if output:
        parts.append(f"{output:,} out")
    modalities = [
        name
        for key, name in (
            ("image_inputs", "image"),
            ("audio_inputs", "audio"),
            ("video_inputs", "video"),
            ("pdf_inputs", "pdf"),
        )
        if profile.get(key)
    ]
    if modalities:
        parts.append("text+" + "+".join(modalities) + " in")
    if profile.get("reasoning_output"):
        parts.append("reasoning")
    if profile.get("tool_calling"):
        parts.append("tools")
    return ", ".join(parts)


def _truncate(rows: list[str]) -> list[str]:
    """Cap a list of bullet rows, appending an ellipsis row when truncated."""
    if len(rows) <= _MAX_ROWS:
        return rows
    hidden = len(rows) - _MAX_ROWS
    return [*rows[:_MAX_ROWS], f"- …and {hidden} more"]


def render_provider_section(provider: str, diff: ProfileDiff) -> str | None:
    """Render the Markdown section for a single provider, or None if unchanged.

    Args:
        provider: Provider identifier (e.g. `'openai'`).
        diff: The computed `ProfileDiff` for the provider.

    Returns:
        Markdown for the provider's changes, or `None` when there are none.
    """
    if diff.is_empty:
        return None

    lines = [f"### {provider}"]

    if diff.added:
        lines.append(f"\n**➕ {len(diff.added)} added**")  # noqa: RUF001
        rows = []
        for model_id in diff.added:
            descriptor = _describe_new_model(diff.added_profiles[model_id])
            suffix = f" — {descriptor}" if descriptor else ""
            rows.append(f"- `{model_id}`{suffix}")
        lines.extend(_truncate(rows))

    if diff.removed:
        lines.append(f"\n**➖ {len(diff.removed)} removed**")  # noqa: RUF001
        lines.extend(_truncate([f"- `{m}`" for m in diff.removed]))

    if diff.changed:
        lines.append(f"\n**✏️ {len(diff.changed)} changed**")
        rows = []
        for model_id, fields in diff.changed.items():
            phrases = [
                _describe_field_change(name, change.old, change.new)
                for name, change in fields.items()
            ]
            rows.append(f"- `{model_id}`: " + "; ".join(phrases))
        lines.extend(_truncate(rows))

    return "\n".join(lines)


def build_summary(provider_diffs: dict[str, ProfileDiff]) -> str:
    """Assemble the full Markdown summary across all providers.

    Args:
        provider_diffs: Mapping of provider name to its `ProfileDiff`.

    Returns:
        Markdown summary. When nothing changed, a short note is returned.
    """
    sections = [
        section
        for provider in sorted(provider_diffs)
        if (section := render_provider_section(provider, provider_diffs[provider]))
    ]
    if not sections:
        return "No model profile data changed."

    total_added = sum(len(d.added) for d in provider_diffs.values())
    total_removed = sum(len(d.removed) for d in provider_diffs.values())
    total_changed = sum(len(d.changed) for d in provider_diffs.values())
    headline = (
        f"**{total_added} added · {total_removed} removed · "
        f"{total_changed} changed** across {len(sections)} provider(s)."
    )

    return "\n\n".join(["## Summary of changes", headline, *sections])


def _verify_ref(repo_root: Path, ref: str) -> None:
    """Confirm `ref` resolves to a commit in `repo_root`.

    Validating once up front lets `_git_show` treat a non-zero exit
    unambiguously as "path absent at this ref", rather than conflating a typo'd
    ref, an unfetched ref, or a non-repository root with a genuinely new file —
    which would otherwise render every existing model as newly added.

    Raises:
        RuntimeError: If git is unavailable, `repo_root` is not a repository, or
            `ref` cannot be resolved.
    """
    try:
        result = subprocess.run(  # noqa: S603
            [  # noqa: S607
                "git",
                "-C",
                str(repo_root),
                "rev-parse",
                "--verify",
                "--quiet",
                f"{ref}^{{commit}}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as e:
        msg = f"Could not run git (is it installed and on PATH?): {e}"
        raise RuntimeError(msg) from e
    if result.returncode != 0:
        msg = (
            f"Could not resolve base ref {ref!r} in {repo_root}; "
            "is it a valid git ref in this repository?"
        )
        raise RuntimeError(msg)


def _git_show(repo_root: Path, ref: str, rel_path: str) -> str | None:
    """Return file contents at `ref`, or None if the file does not exist there.

    Assumes `ref` has already been validated by `_verify_ref`, so a non-zero
    exit here means the path is absent at `ref` rather than a bad ref.
    """
    try:
        result = subprocess.run(  # noqa: S603
            ["git", "-C", str(repo_root), "show", f"{ref}:{rel_path}"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return None
    return result.stdout if result.returncode == 0 else None


def summarize(
    providers: list[ProviderEntry],
    *,
    base_ref: str = "HEAD",
    repo_root: Path | None = None,
) -> str:
    """Build a Markdown summary of profile changes vs `base_ref`.

    Args:
        providers: List of `{'provider': ..., 'data_dir': ...}` entries,
            matching the workflow input. `data_dir` is relative to the repo
            root and contains `_profiles.py`.
        base_ref: Git ref to compare the working tree against.
        repo_root: Repository root. Defaults to the current directory.

    Returns:
        Markdown summary suitable for a PR body.

    Raises:
        RuntimeError: If `base_ref` cannot be resolved or a profiles file exists
            but cannot be read.
        ValueError: If a `providers` entry is missing a required key.
    """
    root = (repo_root or Path.cwd()).resolve()
    _verify_ref(root, base_ref)
    provider_diffs: dict[str, ProfileDiff] = {}

    for entry in providers:
        try:
            provider = entry["provider"]
            data_dir = entry["data_dir"]
        except (KeyError, TypeError) as e:
            msg = (
                f"Invalid provider entry {entry!r}: expected 'provider' and "
                f"'data_dir' keys ({e})"
            )
            raise ValueError(msg) from e
        rel_path = f"{data_dir.rstrip('/')}/_profiles.py"

        old_source = _git_show(root, base_ref, rel_path) or ""
        new_path = root / rel_path
        if new_path.exists():
            try:
                new_source = new_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                msg = f"Could not read {new_path}: {e}"
                raise RuntimeError(msg) from e
        else:
            new_source = ""

        provider_diffs[provider] = diff_profiles(
            extract_profiles(old_source), extract_profiles(new_source)
        )

    return build_summary(provider_diffs)
