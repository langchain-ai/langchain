"""Script for auto-generating api_reference.rst."""
import importlib
import inspect
import typing
from pathlib import Path
from typing import TypedDict, Sequence, List, Dict, Literal, Union
from enum import Enum

from pydantic import BaseModel

ROOT_DIR = Path(__file__).parents[2].absolute()
HERE = Path(__file__).parent

PKG_DIR = ROOT_DIR / "libs" / "langchain" / "langchain"
EXP_DIR = ROOT_DIR / "libs" / "experimental" / "langchain_experimental"
WRITE_FILE = HERE / "api_reference.rst"
EXP_WRITE_FILE = HERE / "experimental_api_reference.rst"


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
    package_directory: Union[str, Path]
) -> Dict[str, ModuleMembers]:
    """Recursively load modules of a package based on the file system.

    Traversal based on the file system makes it easy to determine which
    of the modules/packages are part of the package vs. 3rd party or built-in.

    Parameters:
        package_directory: Path to the package directory.

    Returns:
        list: A list of loaded module objects.
    """
    package_path = (
        Path(package_directory)
        if isinstance(package_directory, str)
        else package_directory
    )
    modules_by_namespace = {}

    package_name = package_path.name

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
            print(f"Error: Unable to import module '{namespace}' with error: {e}")

    return modules_by_namespace


def _construct_doc(pkg: str, members_by_namespace: Dict[str, ModuleMembers]) -> str:
    """Construct the contents of the reference.rst file for the given package.

    Args:
        pkg: The package name
        members_by_namespace: The members of the package, dict organized by top level
                              module contains a list of classes and functions
                              inside of the top level namespace.

    Returns:
        The contents of the reference.rst file.
    """
    full_doc = f"""\
=======================
``{pkg}`` API Reference
=======================

"""
    namespaces = sorted(members_by_namespace)

    for module in namespaces:
        _members = members_by_namespace[module]
        classes = _members["classes_"]
        functions = _members["functions"]
        if not (classes or functions):
            continue
        section = f":mod:`{pkg}.{module}`"
        underline = "=" * (len(section) + 1)
        full_doc += f"""\
{section}
{underline}

.. automodule:: {pkg}.{module}
    :no-members:
    :no-inherited-members:

"""

        if classes:
            full_doc += f"""\
Classes
--------------
.. currentmodule:: {pkg}

.. autosummary::
    :toctree: {module}
"""

            for class_ in sorted(classes, key=lambda c: c["qualified_name"]):
                if not class_["is_public"]:
                    continue

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
            _functions = [f["qualified_name"] for f in functions if f["is_public"]]
            fstring = "\n    ".join(sorted(_functions))
            full_doc += f"""\
Functions
--------------
.. currentmodule:: {pkg}

.. autosummary::
    :toctree: {module}
    :template: function.rst

    {fstring}

"""
    return full_doc


def main() -> None:
    """Generate the reference.rst file for each package."""
    lc_members = _load_package_modules(PKG_DIR)
    lc_doc = ".. _api_reference:\n\n" + _construct_doc("langchain", lc_members)
    with open(WRITE_FILE, "w") as f:
        f.write(lc_doc)
    exp_members = _load_package_modules(EXP_DIR)
    exp_doc = ".. _experimental_api_reference:\n\n" + _construct_doc(
        "langchain_experimental", exp_members
    )
    with open(EXP_WRITE_FILE, "w") as f:
        f.write(exp_doc)


if __name__ == "__main__":
    main()
