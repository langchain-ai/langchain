"""Script for auto-generating api_reference.rst."""

import importlib
import inspect
import os
import typing
from enum import Enum
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

        except ImportError as e:
            print(f"Error: Unable to import module '{namespace}' with error: {e}")  # noqa: T201

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
    package_dir = _package_dir(package_name)
    package_members = _load_package_modules(package_dir)
    package_version = _get_package_version(package_dir)
    with open(_out_file_path(package_name), "w") as f:
        f.write(
            _doc_first_line(package_name)
            + _construct_doc(
                _package_namespace(package_name), package_members, package_version
            )
        )


def _package_namespace(package_name: str) -> str:
    return (
        package_name
        if package_name == "langchain"
        else f"langchain_{package_name.replace('-', '_')}"
    )


def _package_dir(package_name: str = "langchain") -> Path:
    """Return the path to the directory containing the documentation."""
    if package_name in ("langchain", "experimental", "community", "core", "cli"):
        return ROOT_DIR / "libs" / package_name / _package_namespace(package_name)
    else:
        return (
            ROOT_DIR
            / "libs"
            / "partners"
            / package_name
            / _package_namespace(package_name)
        )


def _get_package_version(package_dir: Path) -> str:
    """Return the version of the package."""
    try:
        with open(package_dir.parent / "pyproject.toml", "r") as f:
            pyproject = toml.load(f)
    except FileNotFoundError as e:
        print(
            f"pyproject.toml not found in {package_dir.parent}.\n"
            "You are either attempting to build a directory which is not a package or "
            "the package is missing a pyproject.toml file which should be added."
            "Aborting the build."
        )
        exit(1)
    return pyproject["tool"]["poetry"]["version"]


def _out_file_path(package_name: str) -> Path:
    """Return the path to the file containing the documentation."""
    return HERE / f"{package_name.replace('-', '_')}_api_reference.rst"


def _doc_first_line(package_name: str) -> str:
    """Return the path to the file containing the documentation."""
    return f".. {package_name.replace('-', '_')}_api_reference:\n\n"


def main() -> None:
    """Generate the api_reference.rst file for each package."""
    print("Starting to build API reference files.")
    for dir in os.listdir(ROOT_DIR / "libs"):
        # Skip any hidden directories
        # Some of these could be present by mistake in the code base
        # e.g., .pytest_cache from running tests from the wrong location.
        if dir.startswith("."):
            print("Skipping dir:", dir)
            continue

        if dir in ("cli", "partners"):
            continue
        else:
            print("Building package:", dir)
            _build_rst_file(package_name=dir)
    partner_packages = os.listdir(ROOT_DIR / "libs" / "partners")
    print("Building partner packages:", partner_packages)
    for dir in partner_packages:
        _build_rst_file(package_name=dir)
    print("API reference files built.")


if __name__ == "__main__":
    main()
