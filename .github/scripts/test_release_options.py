"""Verify _release.yml dropdown options match actual package directories.

Dropdown options are short names (e.g. `openai`, `core`). The workflow's
`EFFECTIVE_WORKING_DIR` expression re-adds the `libs/` prefix for top-level
packages and `libs/partners/` for everything else. This test reconstructs the
full path for each short name and compares against packages on disk.
"""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Keep in sync with the non-partner allowlist in `EFFECTIVE_WORKING_DIR`
# in `.github/workflows/_release.yml`.
TOP_LEVEL_PACKAGES = frozenset(
    {"core", "langchain", "langchain_v1", "text-splitters", "standard-tests", "model-profiles"}
)


def _get_release_options() -> list[str]:
    workflow = REPO_ROOT / ".github" / "workflows" / "_release.yml"
    with open(workflow) as f:
        data = yaml.safe_load(f)
    try:
        # PyYAML (YAML 1.1) parses the bare key `on` as boolean True
        return data[True]["workflow_dispatch"]["inputs"]["working-directory"]["options"]
    except (KeyError, TypeError) as e:
        msg = f"Could not find workflow_dispatch options in {workflow}: {e}"
        raise AssertionError(msg) from e


def _expand_option(option: str) -> str:
    if option in TOP_LEVEL_PACKAGES:
        return f"libs/{option}"
    return f"libs/partners/{option}"


def _get_package_dirs() -> set[str]:
    libs = REPO_ROOT / "libs"
    dirs: set[str] = set()
    # Top-level packages (libs/core, libs/langchain, etc.)
    for p in libs.iterdir():
        if p.is_dir() and (p / "pyproject.toml").exists():
            dirs.add(f"libs/{p.name}")
    # Partner packages (libs/partners/*)
    partners = libs / "partners"
    if partners.exists():
        for p in partners.iterdir():
            if p.is_dir() and (p / "pyproject.toml").exists():
                dirs.add(f"libs/partners/{p.name}")
    return dirs


def test_release_options_match_packages() -> None:
    options = {_expand_option(o) for o in _get_release_options()}
    packages = _get_package_dirs()
    missing_from_dropdown = packages - options
    extra_in_dropdown = options - packages
    assert not missing_from_dropdown, (
        f"Packages on disk missing from _release.yml dropdown: {missing_from_dropdown}"
    )
    assert not extra_in_dropdown, (
        f"Dropdown options with no matching package directory: {extra_in_dropdown}"
    )
