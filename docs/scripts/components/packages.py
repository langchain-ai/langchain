import logging
import os
from distutils.sysconfig import get_python_lib
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

ROOT_DIR = Path(os.path.abspath(__file__)).parents[3]
PACKAGE_METADATA_FILE = ROOT_DIR / "packages.yaml"


def get_packages() -> dict[str, str]:
    if not PACKAGE_METADATA_FILE.exists():
        logger.warning(f"The packages file {PACKAGE_METADATA_FILE} does not exist.")
        return {}

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
        package_infos = {}
        for repo in data["repos"]:
            for package in repo["packages"]:
                package_infos[package["name"].replace("-", "_")] = package["path"]

    return package_infos


def get_package_dir(package_name):
    """Get the directory of the package.

    For now, we assume the package is installed in the current environment.
    """
    return Path(get_python_lib()) / package_name.replace("-", "_")


def find_package_files(directory):
    """Recursively find all Python files in the given directory."""
    return (
        p.resolve()
        for p in Path(directory).glob("**/*.py")
        if "tests" not in p.parts
        and "scripts" not in p.parts
        and not p.name.startswith("_")
    )
