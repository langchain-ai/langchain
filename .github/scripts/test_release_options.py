"""Verify _release.yml dropdown options match actual package directories."""

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


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
    options = set(_get_release_options())
    packages = _get_package_dirs()
    missing_from_dropdown = packages - options
    extra_in_dropdown = options - packages
    assert not missing_from_dropdown, (
        f"Packages on disk missing from _release.yml dropdown: {missing_from_dropdown}"
    )
    assert not extra_in_dropdown, (
        f"Dropdown options with no matching package directory: {extra_in_dropdown}"
    )
