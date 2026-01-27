"""CLI for refreshing model profile data from models.dev."""

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import httpx

try:
    import tomllib  # type: ignore[import-not-found]  # Python 3.11+
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


def _validate_data_dir(data_dir: Path) -> Path:
    """Validate and canonicalize data directory path.

    Args:
        data_dir: User-provided data directory path.

    Returns:
        Resolved, canonical path.

    Raises:
        SystemExit: If user declines to write outside current directory.
    """
    # Resolve to absolute, canonical path (follows symlinks)
    try:
        resolved = data_dir.resolve(strict=False)
    except (OSError, RuntimeError) as e:
        msg = f"Invalid data directory path: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    # Warn if writing outside current directory
    cwd = Path.cwd().resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        # Not relative to cwd
        print("⚠️  WARNING: Writing outside current directory", file=sys.stderr)
        print(f"   Current directory: {cwd}", file=sys.stderr)
        print(f"   Target directory:  {resolved}", file=sys.stderr)
        print(file=sys.stderr)
        response = input("Continue? (y/N): ")
        if response.lower() != "y":
            print("Aborted.", file=sys.stderr)
            sys.exit(1)

    return resolved


def _load_augmentations(
    data_dir: Path,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Load augmentations from `profile_augmentations.toml`.

    Args:
        data_dir: Directory containing `profile_augmentations.toml`.

    Returns:
        Tuple of `(provider_augmentations, model_augmentations)`.
    """
    aug_file = data_dir / "profile_augmentations.toml"
    if not aug_file.exists():
        return {}, {}

    try:
        with aug_file.open("rb") as f:
            data = tomllib.load(f)
    except PermissionError:
        msg = f"Permission denied reading augmentations file: {aug_file}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except tomllib.TOMLDecodeError as e:
        msg = f"Invalid TOML syntax in augmentations file: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        msg = f"Failed to read augmentations file: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    overrides = data.get("overrides", {})
    provider_aug: dict[str, Any] = {}
    model_augs: dict[str, dict[str, Any]] = {}

    for key, value in overrides.items():
        if isinstance(value, dict):
            model_augs[key] = value
        else:
            provider_aug[key] = value

    return provider_aug, model_augs


def _model_data_to_profile(model_data: dict[str, Any]) -> dict[str, Any]:
    """Convert raw models.dev data into the canonical profile structure."""
    limit = model_data.get("limit") or {}
    modalities = model_data.get("modalities") or {}
    input_modalities = modalities.get("input") or []
    output_modalities = modalities.get("output") or []

    profile = {
        "max_input_tokens": limit.get("context"),
        "max_output_tokens": limit.get("output"),
        "image_inputs": "image" in input_modalities,
        "audio_inputs": "audio" in input_modalities,
        "pdf_inputs": "pdf" in input_modalities or model_data.get("pdf_inputs"),
        "video_inputs": "video" in input_modalities,
        "image_outputs": "image" in output_modalities,
        "audio_outputs": "audio" in output_modalities,
        "video_outputs": "video" in output_modalities,
        "reasoning_output": model_data.get("reasoning"),
        "tool_calling": model_data.get("tool_call"),
        "tool_choice": model_data.get("tool_choice"),
        "structured_output": model_data.get("structured_output"),
        "image_url_inputs": model_data.get("image_url_inputs"),
        "image_tool_message": model_data.get("image_tool_message"),
        "pdf_tool_message": model_data.get("pdf_tool_message"),
    }

    return {k: v for k, v in profile.items() if v is not None}


def _apply_overrides(
    profile: dict[str, Any], *overrides: dict[str, Any] | None
) -> dict[str, Any]:
    """Merge provider and model overrides onto the canonical profile."""
    merged = dict(profile)
    for override in overrides:
        if not override:
            continue
        for key, value in override.items():
            if value is not None:
                merged[key] = value  # noqa: PERF403
    return merged


def _ensure_safe_output_path(base_dir: Path, output_file: Path) -> None:
    """Ensure the resolved output path remains inside the expected directory."""
    if base_dir.exists() and base_dir.is_symlink():
        msg = f"Data directory {base_dir} is a symlink; refusing to write profiles."
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    if output_file.exists() and output_file.is_symlink():
        msg = (
            f"profiles.py at {output_file} is a symlink; refusing to overwrite it.\n"
            "Delete the symlink or point --data-dir to a safe location."
        )
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    try:
        output_file.resolve(strict=False).relative_to(base_dir.resolve())
    except (OSError, RuntimeError) as e:
        msg = f"Failed to resolve output path: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except ValueError:
        msg = f"Refusing to write outside of data directory: {output_file}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)


def _write_profiles_file(output_file: Path, contents: str) -> None:
    """Write the generated module atomically without following symlinks."""
    _ensure_safe_output_path(output_file.parent, output_file)

    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=output_file.parent, delete=False
        ) as tmp_file:
            tmp_file.write(contents)
            temp_path = Path(tmp_file.name)
        temp_path.replace(output_file)
    except PermissionError:
        msg = f"Permission denied writing file: {output_file}"
        print(f"❌ {msg}", file=sys.stderr)
        if temp_path:
            temp_path.unlink(missing_ok=True)
        sys.exit(1)
    except OSError as e:
        msg = f"Failed to write file: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        if temp_path:
            temp_path.unlink(missing_ok=True)
        sys.exit(1)


MODULE_ADMONITION = """Auto-generated model profiles.

DO NOT EDIT THIS FILE MANUALLY.
This file is generated by the langchain-profiles CLI tool.

It contains data derived from the models.dev project.

Source: https://github.com/sst/models.dev
License: MIT License

To update these data, refer to the instructions here:

https://docs.langchain.com/oss/python/langchain/models#updating-or-overwriting-profile-data
"""


def refresh(provider: str, data_dir: Path) -> None:  # noqa: C901, PLR0915
    """Download and merge model profile data for a specific provider.

    Args:
        provider: Provider ID from models.dev (e.g., `'anthropic'`, `'openai'`).
        data_dir: Directory containing `profile_augmentations.toml` and where
            `profiles.py` will be written.
    """
    # Validate and canonicalize data directory path
    data_dir = _validate_data_dir(data_dir)

    api_url = "https://models.dev/api.json"

    print(f"Provider: {provider}")
    print(f"Data directory: {data_dir}")
    print()

    # Download data from models.dev
    print(f"Downloading data from {api_url}...")
    try:
        response = httpx.get(api_url, timeout=30)
        response.raise_for_status()
    except httpx.TimeoutException:
        msg = f"Request timed out connecting to {api_url}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except httpx.HTTPStatusError as e:
        msg = f"HTTP error {e.response.status_code} from {api_url}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as e:
        msg = f"Failed to connect to {api_url}: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    try:
        all_data = response.json()
    except json.JSONDecodeError as e:
        msg = f"Invalid JSON response from API: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    # Basic validation
    if not isinstance(all_data, dict):
        msg = "Expected API response to be a dictionary"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    provider_count = len(all_data)
    model_count = sum(len(p.get("models", {})) for p in all_data.values())
    print(f"Downloaded {provider_count} providers with {model_count} models")

    # Extract data for this provider
    if provider not in all_data:
        msg = f"Provider '{provider}' not found in models.dev data"
        print(msg, file=sys.stderr)
        sys.exit(1)

    provider_data = all_data[provider]
    models = provider_data.get("models", {})
    print(f"Extracted {len(models)} models for {provider}")

    # Load augmentations
    print("Loading augmentations...")
    provider_aug, model_augs = _load_augmentations(data_dir)

    # Merge and convert to profiles
    profiles: dict[str, dict[str, Any]] = {}
    for model_id, model_data in models.items():
        base_profile = _model_data_to_profile(model_data)
        profiles[model_id] = _apply_overrides(
            base_profile, provider_aug, model_augs.get(model_id)
        )

    # Include new models defined purely via augmentations
    extra_models = set(model_augs) - set(models)
    if extra_models:
        print(f"Adding {len(extra_models)} models from augmentations only...")
    for model_id in sorted(extra_models):
        profiles[model_id] = _apply_overrides({}, provider_aug, model_augs[model_id])

    # Ensure directory exists
    try:
        data_dir.mkdir(parents=True, exist_ok=True, mode=0o755)
    except PermissionError:
        msg = f"Permission denied creating directory: {data_dir}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        msg = f"Failed to create directory: {e}"
        print(f"❌ {msg}", file=sys.stderr)
        sys.exit(1)

    # Write as Python module
    output_file = data_dir / "_profiles.py"
    print(f"Writing to {output_file}...")
    module_content = [f'"""{MODULE_ADMONITION}"""\n', "from typing import Any\n\n"]
    module_content.append("_PROFILES: dict[str, dict[str, Any]] = ")
    json_str = json.dumps(profiles, indent=4)
    json_str = (
        json_str.replace("true", "True")
        .replace("false", "False")
        .replace("null", "None")
    )
    module_content.append(f"{json_str}\n")
    _write_profiles_file(output_file, "".join(module_content))

    print(
        f"✓ Successfully refreshed {len(profiles)} model profiles "
        f"({output_file.stat().st_size:,} bytes)"
    )


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Refresh model profile data from models.dev",
        prog="langchain-profiles",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # refresh command
    refresh_parser = subparsers.add_parser(
        "refresh", help="Download and merge model profile data for a provider"
    )
    refresh_parser.add_argument(
        "--provider",
        required=True,
        help="Provider ID from models.dev (e.g., 'anthropic', 'openai', 'google')",
    )
    refresh_parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Data directory containing profile_augmentations.toml",
    )

    args = parser.parse_args()

    if args.command == "refresh":
        refresh(args.provider, args.data_dir)


if __name__ == "__main__":
    main()
