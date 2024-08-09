"""Partner packages table generator.
Generate a page with a table of the existing partner packages.
"""

import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import requests
import tomli
import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
PACKAGE_METADATA_FILE = ROOT_DIR / "packages.yaml"
DOCS_DIR = ROOT_DIR / "docs" / "docs"
PLATFORMS_FILE = DOCS_DIR / "integrations" / "platforms" / "index.mdx"


@dataclass
class PartnerPackage:
    """Partner package information."""

    name: str
    version: str
    description: str
    url: str
    authors: list[str] = None
    license: str = None


def parse_toml_metadata(toml_content: str) -> PartnerPackage:
    if not toml_content:
        raise ValueError("The metadata content is empty.")
    try:
        metadata = tomli.loads(toml_content)
    except TypeError as e:
        raise ValueError(
            f"Failed to parse the metadata content: {toml_content}. Error: {e}"
        )

    if "tool" not in metadata or "poetry" not in metadata["tool"]:
        raise ValueError("The metadata does not contain the 'tool.poetry' section.")
    tool_poetry = metadata["tool"]["poetry"]
    if "urls" not in tool_poetry or "Source Code" not in tool_poetry["urls"]:
        url = ""
    else:
        url = tool_poetry["urls"]["Source Code"]

    return PartnerPackage(
        name=tool_poetry.get("name", ""),
        version=tool_poetry.get("version", ""),
        description=tool_poetry.get("description", ""),
        authors=tool_poetry.get("authors", []),
        url=url,
        license=tool_poetry.get("license", ""),
    )


def get_toml_content(github_slug: str, package_path: str) -> str:
    local_pyproject_file = ROOT_DIR / Path(package_path) / "pyproject.toml"
    if local_pyproject_file.exists() and github_slug == "langchain-ai/langchain":
        # get the package metadata from the local pyproject.toml files
        with open(local_pyproject_file, "r") as f:
            return f.read()
    else:
        # get the package metadata from the GitHub repository
        url = f"https://raw.githubusercontent.com/{github_slug}/main/{package_path}/pyproject.toml"
        response = requests.get(url)
        if response.status_code != 200:
            raise ValueError(f"The package metadata file {url} does not exist.")
        return response.text


def get_package_metadata(github_slug, package) -> PartnerPackage:
    toml_content = get_toml_content(github_slug, package["path"])
    pack = parse_toml_metadata(toml_content)
    return pack


def get_integration_packages_info() -> list[PartnerPackage]:
    if not PACKAGE_METADATA_FILE.exists():
        logger.warning(f"The packages file {PACKAGE_METADATA_FILE} does not exist.")
        return []

    with open(PACKAGE_METADATA_FILE, "r") as f:
        data = yaml.safe_load(f)

        if data["kind"] != "Package discovery":
            raise ValueError(
                f"The kind of the packages file should be 'Package discovery' but it is {data['kind']}."
            )
        if data["version"] != "v1":
            raise ValueError(
                f"The version of the packages file should be v1 but it is {data['version']}."
            )
        integration_packages = []
        for repo in data["repos"]:
            for package in repo["packages"]:
                if (
                    repo["name"] == "langchain"
                    and "partners" in package["path"]
                    or repo["name"] != "langchain"
                ):
                    package_info = get_package_metadata(repo["slug"], package)
                    integration_packages.append(package_info)

        return integration_packages


def generate_table(packages: list[PartnerPackage]) -> str:
    lines = []
    table_header = """
| Package 🔻 | Version | License | Authors | Description |
|------------|---------|---------|---------|-------------|
"""
    lines.append(table_header)
    for package in sorted(packages, key=lambda x: x.name):
        title_link = f"[{package.name}]({package.url})" if package.url else package.name
        line = " | ".join(
            [
                title_link,
                package.version,
                package.license,
                ", ".join(package.authors),
                package.description,
            ]
        )
        line = f"| {line} |\n"
        lines.append(line)

    logger.info(
        f"Created a table of {len(packages)} lines with partner package information."
    )
    return "".join(lines)


def create_file_with_table(output_file: Path, table_str: str):
    index_file_content = """---
sidebar_position: 0
sidebar_class_name: hidden
---

# Providers

:::info

If you'd like to write your own integration, see [Extending LangChain](/docs/how_to/#custom).
If you'd like to contribute an integration, see [Contributing integrations](/docs/contributing/integrations/).

:::

LangChain integrates with many providers.

## Partner Packages

These providers have separate `langchain-{provider}` packages for improved versioning, dependency management and testing.

{table}


## All Providers

[A full list of all providers](/docs/integrations/providers/).
"""
    index_file_content = index_file_content.replace("{table}", table_str)
    with open(output_file, "w") as f:
        f.write(index_file_content)
        logger.warning(f"{output_file} file updated with the package table.")


def main():
    # extract the package metadata:
    integration_packages = get_integration_packages_info()
    # generate the package table and update a file with the table:
    table_str = generate_table(integration_packages)
    create_file_with_table(PLATFORMS_FILE, table_str)


if __name__ == "__main__":
    main()
