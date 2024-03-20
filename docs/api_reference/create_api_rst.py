"""Script for auto-generating <package>_api_reference.rst files.

Install all packages with `pip install <package> -U> before running this script.
Script parses modules from the local source. If they are not presented locally,
then from the `lib/python3.10/site-packages/<package>` directories.
Some packages are incompatible with other packages with dependencies.
"""

import importlib
import inspect
import os
import sys
import typing
from enum import Enum
from importlib.metadata import metadata
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, TypedDict, Union

import toml
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parents[2].absolute()
HERE = Path(__file__).parent

ClassKind = Literal["TypedDict", "Regular", "Pydantic", "enum"]


class ClassInfo(TypedDict):
    """Information about a class."""

    name: str
    """The name of the class."""
    qualified_name: str
    """The fully qualified name of the class."""
    kind: ClassKind
    """The kind of the class."""
    is_public: bool
    """Whether the class is public or not."""


class FunctionInfo(TypedDict):
    """Information about a function."""

    name: str
    """The name of the function."""
    qualified_name: str
    """The fully qualified name of the function."""
    is_public: bool
    """Whether the function is public or not."""


class ModuleMembers(TypedDict):
    """A dictionary of module members."""

    classes_: Sequence[ClassInfo]
    functions: Sequence[FunctionInfo]


def _load_module_members(module_path: str, namespace: str) -> ModuleMembers:
    """Load all members of a module.

    Args:
        module_path: Path to the module.
        namespace: the namespace of the module.

    Returns:
        list: A list of loaded module objects.
    """
    classes_: List[ClassInfo] = []
    functions: List[FunctionInfo] = []
    module = importlib.import_module(module_path)
    for name, type_ in inspect.getmembers(module):
        if not hasattr(type_, "__module__"):
            continue
        if type_.__module__ != module_path:
            continue

        if inspect.isclass(type_):
            if type(type_) == typing._TypedDictMeta:  # type: ignore
                kind: ClassKind = "TypedDict"
            elif issubclass(type_, Enum):
                kind = "enum"
            elif issubclass(type_, BaseModel):
                kind = "Pydantic"
            else:
                kind = "Regular"

            classes_.append(
                ClassInfo(
                    name=name,
                    qualified_name=f"{namespace}.{name}",
                    kind=kind,
                    is_public=not name.startswith("_"),
                )
            )
        elif inspect.isfunction(type_):
            functions.append(
                FunctionInfo(
                    name=name,
                    qualified_name=f"{namespace}.{name}",
                    is_public=not name.startswith("_"),
                )
            )
        else:
            continue

    return ModuleMembers(
        classes_=classes_,
        functions=functions,
    )


def _merge_module_members(
    module_members: Sequence[ModuleMembers],
) -> ModuleMembers:
    """Merge module members."""
    classes_: List[ClassInfo] = []
    functions: List[FunctionInfo] = []
    for module in module_members:
        classes_.extend(module["classes_"])
        functions.extend(module["functions"])

    return ModuleMembers(
        classes_=classes_,
        functions=functions,
    )


def _load_package_modules(
    package_directory: Union[str, Path], submodule: Optional[str] = None
) -> Dict[str, ModuleMembers]:
    """Recursively load modules of a package based on the file system.

    Traversal based on the file system makes it easy to determine which
    of the modules/packages are part of the package vs. 3rd party or built-in.

    Parameters:
        package_directory: Path to the package directory.
        submodule: Optional name of submodule to load.

    Returns:
        list: A list of loaded module objects.
    """
    package_path = (
        Path(package_directory)
        if isinstance(package_directory, str)
        else package_directory
    )
    modules_by_namespace = {}

    # Get the high level package name
    package_name = package_path.name

    # If we are loading a submodule, add it in
    if submodule is not None:
        package_path = package_path / submodule

    for file_path in package_path.rglob("*.py"):
        if file_path.name.startswith("_"):
            continue

        relative_module_name = file_path.relative_to(package_path)

        # Skip if any module part starts with an underscore
        if any(part.startswith("_") for part in relative_module_name.parts):
            continue

        # Get the full namespace of the module
        namespace = str(relative_module_name).replace(".py", "").replace("/", ".")
        # Keep only the top level namespace
        top_namespace = namespace.split(".")[0]

        try:
            # If submodule is present, we need to construct the paths in a slightly
            # different way
            if submodule is not None:
                module_members = _load_module_members(
                    f"{package_name}.{submodule}.{namespace}",
                    f"{submodule}.{namespace}",
                )
            else:
                module_members = _load_module_members(
                    f"{package_name}.{namespace}", namespace
                )
            # Merge module members if the namespace already exists
            if top_namespace in modules_by_namespace:
                existing_module_members = modules_by_namespace[top_namespace]
                _module_members = _merge_module_members(
                    [existing_module_members, module_members]
                )
            else:
                _module_members = module_members

            modules_by_namespace[top_namespace] = _module_members

        except (ImportError, AttributeError, TypeError) as e:
            print(
                f"Error: Unable to import module '{namespace}' with error: {e}"
            )  # noqa: T201

    return modules_by_namespace


def _construct_doc(
    package_namespace: str,
    members_by_namespace: Dict[str, ModuleMembers],
    package_version: str,
) -> str:
    """Construct the contents of the reference.rst file for the given package.

    Args:
        package_namespace: The package top level namespace
        members_by_namespace: The members of the package, dict organized by top level
                              module contains a list of classes and functions
                              inside of the top level namespace.

    Returns:
        The contents of the reference.rst file.
    """
    full_doc = f"""\
=======================
``{package_namespace}`` {package_version}
=======================

"""
    namespaces = sorted(members_by_namespace)

    for module in namespaces:
        _members = members_by_namespace[module]
        classes = [el for el in _members["classes_"] if el["is_public"]]
        functions = [el for el in _members["functions"] if el["is_public"]]
        if not (classes or functions):
            continue
        section = f":mod:`{package_namespace}.{module}`"
        underline = "=" * (len(section) + 1)
        full_doc += f"""\
{section}
{underline}

.. automodule:: {package_namespace}.{module}
    :no-members:
    :no-inherited-members:

"""

        if classes:
            full_doc += f"""\
Classes
--------------
.. currentmodule:: {package_namespace}

.. autosummary::
    :toctree: {module}
"""

            for class_ in sorted(classes, key=lambda c: c["qualified_name"]):
                if class_["kind"] == "TypedDict":
                    template = "typeddict.rst"
                elif class_["kind"] == "enum":
                    template = "enum.rst"
                elif class_["kind"] == "Pydantic":
                    template = "pydantic.rst"
                else:
                    template = "class.rst"

                full_doc += f"""\
    :template: {template}
    
    {class_["qualified_name"]}
    
"""

        if functions:
            _functions = [f["qualified_name"] for f in functions]
            fstring = "\n    ".join(sorted(_functions))
            full_doc += f"""\
Functions
--------------
.. currentmodule:: {package_namespace}

.. autosummary::
    :toctree: {module}
    :template: function.rst

    {fstring}

"""
    return full_doc


def _build_rst_file(package_name: str = "langchain") -> None:
    """Create a rst file for building of documentation.

    Args:
        package_name: Can be either "langchain" or "core" or "experimental".
    """
    package_dir = _get_package_dir(package_name)
    package_namespace = _get_package_namespace(package_name)
    package_dir = _manage_external_packages(package_dir, package_namespace)
    print("  Package path:", package_dir)
    package_version = _get_package_version(package_dir, package_namespace)
    print("  version:", package_version)
    package_members = _load_package_modules(package_dir)
    print("  modules:", len(package_members))
    if package_members:
        # do not recreate the file if there are no members
        with open(_out_file_path(package_name), "w") as f:
            f.write(
                _doc_first_line(package_name)
                + _construct_doc(package_namespace, package_members, package_version)
            )
            print(f"  Created: {_out_file_path(package_name)}")
    else:
        print(
            f"  NOT created: {_out_file_path(package_name)}, since no members found for package '{package_name}'."
        )


def _get_package_namespace(package_name: str) -> str:
    return (
        package_name
        if package_name == "langchain"
        else f"langchain_{package_name.replace('-', '_')}"
    )


def _get_package_dir(package_name: str = "langchain") -> Path:
    """Return the path to the directory containing the documentation."""
    if package_name in (
        "langchain",
        "experimental",
        "community",
        "core",
        "cli",
        "text-splitters",
    ):
        return ROOT_DIR / "libs" / package_name / _get_package_namespace(package_name)
    else:
        return (
            ROOT_DIR
            / "libs"
            / "partners"
            / package_name
            / _get_package_namespace(package_name)
        )


def _get_package_version(package_dir: Path, package_name: str) -> str:
    """Return the version of the package."""
    if _is_external_package(package_dir):
        md = metadata(package_name)
        if "Version" in md:
            return md["Version"]
        elif "version" in md:
            return md["version"]
        else:
            raise ValueError(
                f"Error: Unable to find the version of the package '{package_name}'."
            )
    else:
        try:
            with open(package_dir.parent / "pyproject.toml", "r") as f:
                pyproject = toml.load(f)
        except FileNotFoundError as e:
            raise ValueError(
                f"pyproject.toml not found in {package_dir.parent}.\n"
                "You are either attempting to build a directory which is not a package or "
                "the package is missing a pyproject.toml file which should be added."
                "Aborting the build."
            )
        return pyproject["tool"]["poetry"]["version"]


def _is_external_package(package_dir: Path) -> bool:
    """Return True if the package is in the external repo."""
    return not (package_dir.parent / "pyproject.toml").exists()


def _manage_external_packages(package_dir: Path, package_name: str) -> Path:
    """Manage the packages in external repo.

    If the package is in the external repo, then we assume that the package_dir
    does not keep the 'pyproject.toml' file. We assume that the package was
    pip-installed and placed in the 'site-packages' directory.
    """
    if _is_external_package(package_dir):
        try:
            module = importlib.import_module(package_name)
        except (ModuleNotFoundError, ImportError, TypeError) as e:
            raise ImportError(
                f"Error: Unable to import package '{package_name}' with error: {e} "
                f"Use 'pip install {package_name} -U' to install the package."
            )
        if isinstance(module.__path__, list) and module.__path__:
            package_dir = Path(module.__path__[0])
        else:
            raise ImportError(
                f"Error: Unable to find the path to the package '{package_name}'."
            )
    return package_dir


def _out_file_path(package_name: str) -> Path:
    """Return the path to the file containing the documentation."""
    return HERE / f"{package_name.replace('-', '_')}_api_reference.rst"


def _doc_first_line(package_name: str) -> str:
    """Return the path to the file containing the documentation."""
    return f".. {package_name.replace('-', '_')}_api_reference:\n\n"


def main(dirs: Optional[list] = None) -> None:
    """Generate the api_reference.rst file for each package."""
    print("Starting to build API reference files.")
    if not dirs:
        dirs = [
            dir_
            for dir_ in os.listdir(ROOT_DIR / "libs")
            if dir_ not in ("cli", "partners")
        ]
        dirs += os.listdir(ROOT_DIR / "libs" / "partners")
    for dir_ in sorted(dirs):
        # Skip any hidden directories
        # Some of these could be present by mistake in the code base
        # e.g., .pytest_cache from running tests from the wrong location.
        if dir_.startswith("."):
            print("Skipping dir:", dir_)
            continue
        else:
            print("\nBuilding package:", dir_)
            try:
                _build_rst_file(package_name=dir_)
            except ImportError as e:
                print(f"  {e}")
                continue

    print("\nAPI reference files built.\n")


if __name__ == "__main__":
    dirs = sys.argv[1:] or None
    main(dirs=dirs)
