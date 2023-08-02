"""Script for auto-generating api_reference.rst."""
import glob
import importlib
import inspect
import re
import typing

from pathlib import Path
from typing import TypedDict, Sequence, List, Dict, Literal, Union

from pydantic import BaseModel

ROOT_DIR = Path(__file__).parents[2].absolute()
HERE = Path(__file__).parent

PKG_DIR = ROOT_DIR / "libs" / "langchain" / "langchain"
EXP_DIR = ROOT_DIR / "libs" / "experimental" / "langchain_experimental"
WRITE_FILE = HERE / "api_reference.rst"
EXP_WRITE_FILE = HERE / "experimental_api_reference.rst"


ClassKind = Literal["TypedDict", "Regular", "Pydantic"]


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
        if not file_path.name.startswith("__"):
            relative_module_name = file_path.relative_to(package_path)
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


# def load_members(dir: Path) -> dict:
#     members: dict = {}
#     for py in glob.glob(str(dir) + "/**/*.py", recursive=True):
#         module = py[len(str(dir)) + 1 :].replace(".py", "").replace("/", ".")
#         top_level = module.split(".")[0]
#         if top_level not in members:
#             members[top_level] = {"classes": [], "functions": []}
#         with open(py, "r") as f:
#             for line in f.readlines():
#                 cls = re.findall(r"^class ([^_].*)\(", line)
#                 members[top_level]["classes"].extend([module + "." + c for c in cls])
#                 func = re.findall(r"^def ([^_].*)\(", line)
#                 afunc = re.findall(r"^async def ([^_].*)\(", line)
#                 func_strings = [module + "." + f for f in func + afunc]
#                 members[top_level]["functions"].extend(func_strings)
#     return members
#


def _construct_doc(pkg: str, members_by_namespace: Dict[str, ModuleMembers]) -> str:
    """Construct the contents of the reference.rst file for the given package.

    Args:
        pkg: The package name
        members_by_namespace: The members of the package, dict organized by top level module.
                 contains a list of classes and functions inside of the top level
                 namespace

    Returns:
        The contents of the reference.rst file.
    """
    full_doc = f"""\
=============
``{pkg}`` API Reference
=============

"""
    namespaces = sorted(members_by_namespace)

    for module in namespaces:
        _members = members_by_namespace[module]
        raise ValueError(_members)
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

        typed_dicts = [c for c in classes if c["kind"] == "TypedDict"]
        regular_or_pydantic = [
            c for c in classes if c["kind"] in {"Regular", "Pydantic"}
        ]

        if regular_or_pydantic:
            cstring = "\n    ".join(sorted(regular_or_pydantic))
            full_doc += f"""\
Classes
--------------
.. currentmodule:: {pkg}

.. autosummary::
    :toctree: {module}
    :template: class.rst

    {cstring}

"""
        if typed_dicts:
            cstring = "\n    ".join(sorted(typed_dicts))
            full_doc += f"""\
TypedDicts
--------------
.. currentmodule:: {pkg}

.. autosummary::
    :toctree: {module}
    :template: typeddict.rst
    
    {cstring}
"""

        if functions:
            fstring = "\n    ".join(sorted(functions))
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
    # exp_members = load_members(EXP_DIR)
    # exp_doc = ".. _experimental_api_reference:\n\n" + construct_doc(
    #     "langchain_experimental", exp_members
    # )
    # with open(EXP_WRITE_FILE, "w") as f:
    #     f.write(exp_doc)


if __name__ == "__main__":
    main()
