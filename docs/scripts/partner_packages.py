"""Partner packages table generator.
Generate a page with a table of the existing partner packages.
"""

import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import tomli
import yaml

logger = logging.getLogger(__name__)

_ROOT_DIR = Path(os.path.abspath(__file__)).parents[2]
DOCS_DIR = _ROOT_DIR / "docs" / "docs"
OUTPUT_DIR = DOCS_DIR / "additional_resources"
CODE_DIR = _ROOT_DIR / "libs"
PARTNER_PACKAGES_DIR = CODE_DIR / "partners"
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
    external_dependencies: list[str] = None


def get_partner_dirs(partner_packages_dir: Path) -> list[Path]:
    """Get the partner package directories."""
    partner_dirs = []
    for partner_dir in partner_packages_dir.iterdir():
        if partner_dir.is_dir():
            partner_dirs.append(partner_dir)
    return sorted(partner_dirs)


def parse_toml_metadata(metadata):
    if "tool" not in metadata or "poetry" not in metadata["tool"]:
        raise ValueError("The metadata does not contain the 'tool.poetry' section.")
    tool_poetry = metadata["tool"]["poetry"]
    name = tool_poetry["name"] if "name" in tool_poetry else ""
    version = tool_poetry["version"] if "version" in tool_poetry else ""
    description = tool_poetry["description"] if "description" in tool_poetry else ""
    authors = tool_poetry["authors"] if "authors" in tool_poetry else []
    license_ = tool_poetry["license"] if "license" in tool_poetry else ""
    if "urls" not in tool_poetry or "Source Code" not in tool_poetry["urls"]:
        url = ""
    else:
        url = tool_poetry["urls"]["Source Code"]

    return PartnerPackage(
        name=name,
        version=version,
        description=description,
        authors=authors,
        url=url,
        license=license_,
    )


def get_external_package_info():
    external_package_info_file = PARTNER_PACKAGES_DIR / "external_packages.yaml"
    if not external_package_info_file.exists():
        logger.warning(
            f"The external packages file {external_package_info_file} does not exist."
        )
        return []

    with open(external_package_info_file, "r") as f:
        data = yaml.safe_load(f)

        if data["kind"] != "Package discovery":
            raise ValueError(
                f"The kind of the external packages file should be 'Package discovery' but it is {data['kind']}."
            )
        if data["version"] != "v1":
            raise ValueError(
                f"The version of the external packages file should be v1 but it is {data['version']}."
            )

        external_packages = [
            PartnerPackage(
                name=package["name"],
                version=package["version"],
                description=package.get("description", ""),
                authors=package.get("authors", []),
                url=package.get("source_code", ""),
                license=package.get("license", ""),
            )
            for package in data["external_packages"]
        ]
        return external_packages


def _compound_partner_metadatas(partner_metadatas, external_packages):
    # check for duplicate package names
    partner_metadatas_names = [partner.name for partner in partner_metadatas]
    external_packages_names = [package.name for package in external_packages]
    if names_in_both := set(partner_metadatas_names) & set(external_packages_names):
        raise ValueError(f"Duplicate package names: {names_in_both}")
    return partner_metadatas + external_packages


def get_partner_metadatas(partner_dirs: list[Path]) -> list[PartnerPackage]:
    """Get the partner package metadata."""
    # parse the metadata from the `partners/{partner}/pyproject.toml` files
    partner_metadatas = []
    for partner_dir in partner_dirs:
        metadata_file = partner_dir / "pyproject.toml"
        module_name = f"langchain_{partner_dir.name}"
        partner_package = None
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                content = f.read()
                metadata = tomli.loads(content)
                partner_package = parse_toml_metadata(metadata)

        if not partner_package:
            logger.warning(
                f"langchain-{partner_dir.name} package: Metadata for the package not found."
            )
            continue

        partner_metadatas.append(partner_package)

    # parse the metadata for external packages
    external_packages = get_external_package_info()

    # compound the metadata
    all_partner_metadatas = _compound_partner_metadatas(
        partner_metadatas, external_packages
    )
    return all_partner_metadatas


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
    # search the partner package directories:
    partner_dirs = get_partner_dirs(PARTNER_PACKAGES_DIR)

    # extract the package metadata:
    partner_metadatas = get_partner_metadatas(partner_dirs)

    # generate the package table and update a file with the table:
    table_str = generate_table(partner_metadatas)
    create_file_with_table(PLATFORMS_FILE, table_str)


if __name__ == "__main__":
    main()
