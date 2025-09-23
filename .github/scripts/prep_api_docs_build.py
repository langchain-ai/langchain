#!/usr/bin/env python
"""Sync libraries from various repositories into this monorepo."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict

import yaml


def load_packages_yaml() -> Dict[str, Any]:
    """Load and parse the packages.yml file."""
    with open("langchain/libs/packages.yml", "r") as f:
        return yaml.safe_load(f)


def get_target_dir(package_name: str) -> Path:
    """Get the target directory for a given package."""
    package_name_short = package_name.replace("langchain-", "")
    base_path = Path("langchain/libs")
    if package_name_short == "experimental":
        return base_path / "experimental"
    if package_name_short == "community":
        return base_path / "community"
    return base_path / "partners" / package_name_short


def clean_target_directories(packages: list) -> None:
    """Remove old directories that will be replaced."""
    for package in packages:
        target_dir = get_target_dir(package["name"])
        if target_dir.exists():
            print(f"Removing {target_dir}")
            shutil.rmtree(target_dir)


def move_libraries(packages: list) -> None:
    """Move libraries from their source locations to the target directories."""
    for package in packages:
        repo_name = package["repo"].split("/")[1]
        source_path = package["path"]
        target_dir = get_target_dir(package["name"])

        # Handle root path case
        if source_path == ".":
            source_dir = repo_name
        else:
            source_dir = f"{repo_name}/{source_path}"

        print(f"Moving {source_dir} to {target_dir}")

        # Ensure target directory exists
        os.makedirs(os.path.dirname(target_dir), exist_ok=True)

        try:
            # Move the directory
            shutil.move(source_dir, target_dir)
        except Exception as e:
            print(f"Error moving {source_dir} to {target_dir}: {e}")


def main():
    """Main function to orchestrate the library sync process."""
    try:
        # Load packages configuration
        package_yaml = load_packages_yaml()

        # Clean target directories
        clean_target_directories(
            [
                p
                for p in package_yaml["packages"]
                if (
                    p["repo"].startswith("langchain-ai/") or p.get("include_in_api_ref")
                )
                and p["repo"] != "langchain-ai/langchain"
                and p["name"]
                != "langchain-ai21"  # Skip AI21 due to dependency conflicts
            ]
        )

        # Move libraries to their new locations
        move_libraries(
            [
                p
                for p in package_yaml["packages"]
                if not p.get("disabled", False)
                and (
                    p["repo"].startswith("langchain-ai/") or p.get("include_in_api_ref")
                )
                and p["repo"] != "langchain-ai/langchain"
                and p["name"]
                != "langchain-ai21"  # Skip AI21 due to dependency conflicts
            ]
        )

        # Delete ones without a pyproject.toml
        for partner in Path("langchain/libs/partners").iterdir():
            if partner.is_dir() and not (partner / "pyproject.toml").exists():
                print(f"Removing {partner} as it does not have a pyproject.toml")
                shutil.rmtree(partner)

        print("Library sync completed successfully!")

    except Exception as e:
        print(f"Error during library sync: {e}")
        raise


if __name__ == "__main__":
    main()
