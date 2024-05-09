"""Partner packages table generator.
Generate a page with a table of the existing partner packages.
"""

import importlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import tomli

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
    external_dependencies: list[str] = None
    api_ref_link: str = None
    implemented_component2member_links: dict[str, dict[set[str]]] = None


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
    )


def is_module_loaded(module_name):
    try:
        importlib.import_module(module_name)
        return True
    except ModuleNotFoundError:
        return False


def get_module_info(module_name: str) -> PartnerPackage:
    """Retrieve meta-information about a loaded module/package.

    Args:
      module_name: The name of the loaded module/package.

    Returns:
        PartnerPackage: The meta information about the module/package.
    """
    package_dist = importlib.metadata.distribution(module_name)
    if hasattr(package_dist, "authors"):
        authors = package_dist.authors
    else:
        authors = []
    if (
        hasattr(package_dist, "metadata")
        and hasattr(package_dist.metadata, "json")
        and "project_url" in package_dist.metadata.json
        and len(package_dist.metadata.json["project_url"]) > 1
        and ", " in package_dist.metadata.json["project_url"][1]
    ):
        url = package_dist.metadata.json["project_url"][1].split(",")[1].strip()
    else:
        url = ""
    info = PartnerPackage(
        name=package_dist.name,
        version=package_dist.version,
        description=f"{package_dist.metadata['summary']}",
        url=url,
        authors=authors,
    )
    return info


def get_partner_metadatas(partner_dirs: list[Path]) -> list[PartnerPackage]:
    """Get the partner package metadata."""
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
        # parse the module metadata, if module is loaded as an external package
        if not partner_package and is_module_loaded(module_name):
            partner_package = get_module_info(module_name)

        if not partner_package:
            logger.warning(
                f"langchain-{partner_dir.name} package: Metadata for the package not found."
            )
            continue

        partner_metadatas.append(partner_package)
    return partner_metadatas


def generate_table(packages: list[PartnerPackage]) -> str:
    lines = []
    table_header = """
| Package 🔻 | Version | Authors | Description |
|------------------|---------|-------------------|-------------------------|
"""
    lines.append(table_header)
    for package in sorted(packages, key=lambda x: x.name):
        title_link = f"[{package.name}]({package.url})" if package.url else package.name
        line = " | ".join(
            [
                title_link,
                package.version,
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

These providers have standalone `langchain-{provider}` packages for improved versioning, dependency management and testing.

{table}

## Featured Community Providers

- [Hugging Face](/docs/integrations/platforms/huggingface)
- [Microsoft](/docs/integrations/platforms/microsoft)

## All Providers

Click [here](/docs/integrations/providers/) to see all providers.
"""
    index_file_content = index_file_content.replace("{table}", table_str)
    with open(output_file, "w") as f:
        f.write(index_file_content)
        logger.info(f"{output_file} file updated with the package table.")


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
