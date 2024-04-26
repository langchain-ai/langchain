"""Generate migrations from langchain to langchain-community or core packages."""
import glob
from pathlib import Path
from typing import List, Tuple

from langchain_cli.namespaces.migrate.generate.utils import (
    _get_current_module,
    find_imports_from_package,
)

HERE = Path(__file__).parent
PKGS_ROOT = HERE.parent.parent.parent
LANGCHAIN_PKG = PKGS_ROOT / "langchain"
COMMUNITY_PKG = PKGS_ROOT / "community"
PARTNER_PKGS = PKGS_ROOT / "partners"


def _generate_migrations_from_file(
    source_module: str, code: str, *, from_package: str
) -> List[Tuple[str, str]]:
    """Generate migrations"""
    imports = find_imports_from_package(code, from_package=from_package)
    return [
        # Rewrite in a list comprehension
        (f"{source_module}.{item}", f"{new_path}.{item}")
        for new_path, item in imports
    ]


def _generate_migrations_from_file_in_pkg(
    file: str, root_pkg: str
) -> List[Tuple[str, str]]:
    """Generate migrations for a file that's relative to langchain pkg."""
    # Read the file.
    with open(file, encoding="utf-8") as f:
        code = f.read()

    module_name = _get_current_module(file, root_pkg)
    return _generate_migrations_from_file(
        module_name, code, from_package="langchain_community"
    )


def generate_migrations_from_langchain_to_community() -> List[Tuple[str, str]]:
    """Generate migrations from langchain to langchain-community."""
    migrations = []
    # scanning files in pkg
    for file_path in glob.glob(str(LANGCHAIN_PKG) + "**/*.py"):
        migrations.extend(
            _generate_migrations_from_file_in_pkg(file_path, str(LANGCHAIN_PKG))
        )
    return migrations
