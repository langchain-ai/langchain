import logging
import os
from pathlib import Path
from typing import Optional

from docs.scripts.components.kind import ComponentKind
from docs.scripts.components.metadata import ComponentMetadata, component_metadatas
from docs.scripts.components.modules import (
    find_classes_in_file,
    get_class_namespace,
    is_derived,
)
from docs.scripts.components.packages import (
    find_package_files,
    get_package_dir,
    get_packages,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

_ROOT_DIR = Path(os.path.abspath(__file__)).parents[3]
DOCS_DIR = _ROOT_DIR / "docs" / "docs"
CODE_DIR = _ROOT_DIR / "libs"
LANGCHAIN_PYTHON_URL = "python.langchain.com"


def generate_table(class_fully_qualified_names: list[str]) -> str:
    lines = []
    table_header = """
| Namespace 🔻 | Class |
|------------|---------|
"""
    lines.append(table_header)
    for class_fully_qualified_name in sorted(class_fully_qualified_names):
        class_link = get_class_link(class_fully_qualified_name)
        line = " | ".join(
            [
                ".".join(class_fully_qualified_name.split(".")[:-1]),
                class_link,
            ]
        )
        lines.append(f"| {line} |\n")

    logger.info(
        f"Created a table of {len(class_fully_qualified_names)} lines with component information."
    )
    return "".join(lines)


def _format_api_ref_url(class_fully_qualified_name: str) -> str:
    second_level_namespace = class_fully_qualified_name.split(".")[1]
    return f"https://api.{LANGCHAIN_PYTHON_URL}/en/latest/{second_level_namespace}/{class_fully_qualified_name}.html"


def get_class_link(class_fully_qualified_name: str) -> str:
    short_name = class_fully_qualified_name.split(".")[-1]
    return f"[{short_name}]({_format_api_ref_url(class_fully_qualified_name)})"


def generate_component_index_page(
    component_metadata: ComponentMetadata,
    docs_dir: Path,
    class_fully_qualified_names: list[str],
):
    if not class_fully_qualified_names:
        logger.warning(
            f"No derived classes found for component '{component_metadata.kind}' "
            f"with base class '{component_metadata.base_class}'"
        )
        return
    class_fully_qualified_names = sorted(class_fully_qualified_names)
    logger.info(f"\t{len(class_fully_qualified_names) = }")
    component_dir = docs_dir / "integrations" / component_metadata.kind.value.lower()
    if not component_dir.exists():
        logger.warning(
            f"Component directory '{component_dir}' does not exist. Skipping..."
        )
        return

    output_file = component_dir / "index.mdx"
    table_str = generate_table(class_fully_qualified_names)
    component_name = component_metadata.kind.name.replace("_", " ").title()
    # patch for the 'embedding' component
    if component_name == "Embedding":
        component_name = "Embedding model"
    base_class_link = get_class_link(component_metadata.base_class)

    index_file_content = """---
sidebar_position: 0
sidebar_class_name: hidden
---

# {component_name}

**{component_name}** classes are implemented by inheriting the {base_class_link} class.

This table lists all {classes_num} derived classes.

{table}"""
    index_file_content = (
        index_file_content.replace("{component_name}", component_name)
        .replace("{classes_num}", str(len(class_fully_qualified_names)))
        .replace("{base_class_link}", base_class_link)
        .replace("{table}", table_str)
    )
    with open(output_file, "w") as f:
        f.write(index_file_content)
        logger.warning(
            f"{output_file} file created for the {component_name} component with {len(class_fully_qualified_names)} classes."
        )


def main(component_names: Optional[list[str]] = None):
    """Generate the index page for each component.
    It generates a table of derived classes from the base class of the component.

    It searches the component classes with pip-installed packages only.
    """
    packages = get_packages()
    exclude_packages = ["langchain_cli", "langchain_standard_tests"]
    # index page for these components created manually
    exclude_components = [
        ComponentKind.LLM,
        ComponentKind.CHAT_MODEL,
        ComponentKind.DOCUMENT_LOADER,
    ]

    processing_components = (
        component_metadatas
        if not component_names
        else [
            c
            for c in component_metadatas
            if c.kind.name in [n.upper() for n in component_names]
        ]
    )
    for component_metadata in processing_components:
        if component_metadata.kind in exclude_components:
            logger.warning(
                f"Component '{component_metadata.kind}' is excluded from processing."
            )
            continue
        logger.info(
            f"\n{component_metadata.kind.name} : {component_metadata.base_class}  ================================= "
        )
        all_class_fqns = []
        for package_name, package_path in packages.items():
            if package_name in exclude_packages:
                continue
            for file_path in find_package_files(get_package_dir(package_name)):
                class_namespace = get_class_namespace(file_path, package_name)
                class_names = find_classes_in_file(file_path)
                # remove private classes
                class_fqns = [
                    f"{class_namespace}.{class_name}"
                    for class_name in class_names
                    if not class_name.startswith("_")
                ]
                all_class_fqns += [
                    class_fqn
                    for class_fqn in class_fqns
                    if is_derived(component_metadata.base_class, class_fqn)
                ]

        generate_component_index_page(
            component_metadata,
            docs_dir=DOCS_DIR,
            class_fully_qualified_names=all_class_fqns,
        )


if __name__ == "__main__":
    main(component_names=["embedding"])
    # main()
