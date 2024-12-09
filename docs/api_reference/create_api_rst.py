"""Script for auto-generating api_reference.rst."""

import importlib
import inspect
import os
import sys
import typing
from enum import Enum
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, TypedDict, Union

import toml
import typing_extensions
from langchain_core.runnables import Runnable, RunnableSerializable
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parents[2].absolute()
HERE = Path(__file__).parent

ClassKind = Literal[
    "TypedDict",
    "Regular",
    "Pydantic",
    "enum",
    "RunnablePydantic",
    "RunnableNonPydantic",
]


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
    is_deprecated: bool
    """Whether the class is deprecated."""


class FunctionInfo(TypedDict):
    """Information about a function."""

    name: str
    """The name of the function."""
    qualified_name: str
    """The fully qualified name of the function."""
    is_public: bool
    """Whether the function is public or not."""
    is_deprecated: bool
    """Whether the function is deprecated."""


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

    if ":private:" in (module.__doc__ or ""):
        return ModuleMembers(classes_=[], functions=[])

    for name, type_ in inspect.getmembers(module):
        if not hasattr(type_, "__module__"):
            continue
        if type_.__module__ != module_path:
            continue
        if ":private:" in (type_.__doc__ or ""):
            continue

        if inspect.isclass(type_):
            # The type of the class is used to select a template
            # for the object when rendering the documentation.
            # See `templates` directory for defined templates.
            # This is a hacky solution to distinguish between different
            # kinds of thing that we want to render.
            if type(type_) is typing_extensions._TypedDictMeta:  # type: ignore
                kind: ClassKind = "TypedDict"
            elif type(type_) is typing._TypedDictMeta:  # type: ignore
                kind: ClassKind = "TypedDict"
            elif (
                issubclass(type_, Runnable)
                and issubclass(type_, BaseModel)
                and type_ is not Runnable
            ):
                # RunnableSerializable subclasses from Pydantic which
                # for which we use autodoc_pydantic for rendering.
                # We need to distinguish these from regular Pydantic
                # classes so we can hide inherited Runnable methods
                # and provide a link to the Runnable interface from
                # the template.
                kind = "RunnablePydantic"
            elif (
                issubclass(type_, Runnable)
                and not issubclass(type_, BaseModel)
                and type_ is not Runnable
            ):
                # These are not pydantic classes but are Runnable.
                # We'll hide all the inherited methods from Runnable
                # but use a regular class template to render.
                kind = "RunnableNonPydantic"
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
                    is_deprecated=".. deprecated::" in (type_.__doc__ or ""),
                )
            )
        elif inspect.isfunction(type_):
            functions.append(
                FunctionInfo(
                    name=name,
                    qualified_name=f"{namespace}.{name}",
                    is_public=not name.startswith("_"),
                    is_deprecated=".. deprecated::" in (type_.__doc__ or ""),
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
        package_directory (Union[str, Path]): Path to the package directory.
        submodule (Optional[str]): Optional name of submodule to load.

    Returns:
        Dict[str, ModuleMembers]: A dictionary where keys are module names and values are ModuleMembers objects.
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
            print(f"Error: Unable to import module '{namespace}' with error: {e}")

    return modules_by_namespace


def _construct_doc(
    package_namespace: str,
    members_by_namespace: Dict[str, ModuleMembers],
    package_version: str,
) -> List[typing.Tuple[str, str]]:
    """Construct the contents of the reference.rst file for the given package.

    Args:
        package_namespace: The package top level namespace
        members_by_namespace: The members of the package, dict organized by top level
                              module contains a list of classes and functions
                              inside of the top level namespace.

    Returns:
        The contents of the reference.rst file.
    """
    docs = []
    index_doc = f"""\
:html_theme.sidebar_secondary.remove:

.. currentmodule:: {package_namespace}

.. _{package_namespace}:

======================================
{package_namespace.replace('_', '-')}: {package_version}
======================================

.. automodule:: {package_namespace}
    :no-members:
    :no-inherited-members:

.. toctree::
    :hidden:
    :maxdepth: 2
    
"""
    index_autosummary = """
"""
    namespaces = sorted(members_by_namespace)

    for module in namespaces:
        index_doc += f"    {module}\n"
        module_doc = f"""\
.. currentmodule:: {package_namespace}

.. _{package_namespace}_{module}:
"""
        _members = members_by_namespace[module]
        classes = [
            el
            for el in _members["classes_"]
            if el["is_public"] and not el["is_deprecated"]
        ]
        functions = [
            el
            for el in _members["functions"]
            if el["is_public"] and not el["is_deprecated"]
        ]
        deprecated_classes = [
            el for el in _members["classes_"] if el["is_public"] and el["is_deprecated"]
        ]
        deprecated_functions = [
            el
            for el in _members["functions"]
            if el["is_public"] and el["is_deprecated"]
        ]
        if not (classes or functions):
            continue
        section = f":mod:`{module}`"
        underline = "=" * (len(section) + 1)
        module_doc += f"""
{section}
{underline}

.. automodule:: {package_namespace}.{module}
    :no-members:
    :no-inherited-members:

"""

        index_autosummary += f"""
:ref:`{package_namespace}_{module}`
{'^' * (len(package_namespace) + len(module) + 8)}
"""

        if classes:
            module_doc += f"""\
**Classes**

.. currentmodule:: {package_namespace}

.. autosummary::
    :toctree: {module}
"""
            index_autosummary += """
**Classes**

.. autosummary::
"""

            for class_ in sorted(classes, key=lambda c: c["qualified_name"]):
                if class_["kind"] == "TypedDict":
                    template = "typeddict.rst"
                elif class_["kind"] == "enum":
                    template = "enum.rst"
                elif class_["kind"] == "Pydantic":
                    template = "pydantic.rst"
                elif class_["kind"] == "RunnablePydantic":
                    template = "runnable_pydantic.rst"
                elif class_["kind"] == "RunnableNonPydantic":
                    template = "runnable_non_pydantic.rst"
                else:
                    template = "class.rst"

                module_doc += f"""\
    :template: {template}
    
    {class_["qualified_name"]}
    
"""
                index_autosummary += f"""
    {class_['qualified_name']}
"""

        if functions:
            _functions = [f["qualified_name"] for f in functions]
            fstring = "\n    ".join(sorted(_functions))
            module_doc += f"""\
**Functions**

.. currentmodule:: {package_namespace}

.. autosummary::
    :toctree: {module}
    :template: function.rst

    {fstring}

"""

            index_autosummary += f"""
**Functions**

.. autosummary::

    {fstring}
"""
        if deprecated_classes:
            module_doc += f"""\
**Deprecated classes**

.. currentmodule:: {package_namespace}

.. autosummary::
    :toctree: {module}
"""

            index_autosummary += """
**Deprecated classes**

.. autosummary::
"""

            for class_ in sorted(deprecated_classes, key=lambda c: c["qualified_name"]):
                if class_["kind"] == "TypedDict":
                    template = "typeddict.rst"
                elif class_["kind"] == "enum":
                    template = "enum.rst"
                elif class_["kind"] == "Pydantic":
                    template = "pydantic.rst"
                elif class_["kind"] == "RunnablePydantic":
                    template = "runnable_pydantic.rst"
                elif class_["kind"] == "RunnableNonPydantic":
                    template = "runnable_non_pydantic.rst"
                else:
                    template = "class.rst"

                module_doc += f"""\
    :template: {template}

    {class_["qualified_name"]}

"""
                index_autosummary += f"""
    {class_['qualified_name']}
"""

        if deprecated_functions:
            _functions = [f["qualified_name"] for f in deprecated_functions]
            fstring = "\n    ".join(sorted(_functions))
            module_doc += f"""\
**Deprecated functions**

.. currentmodule:: {package_namespace}

.. autosummary::
    :toctree: {module}
    :template: function.rst

    {fstring}

"""
            index_autosummary += f"""
**Deprecated functions**

.. autosummary::

    {fstring}

"""
        docs.append((f"{module}.rst", module_doc))
    docs.append(("index.rst", index_doc + index_autosummary))
    return docs


def _build_rst_file(package_name: str = "langchain") -> None:
    """Create a rst file for building of documentation.

    Args:
        package_name: Can be either "langchain" or "core" or "experimental".
    """
    package_dir = _package_dir(package_name)
    package_members = _load_package_modules(package_dir)
    package_version = _get_package_version(package_dir)
    output_dir = _out_file_path(package_name)
    os.mkdir(output_dir)
    rsts = _construct_doc(
        _package_namespace(package_name), package_members, package_version
    )
    for name, rst in rsts:
        with open(output_dir / name, "w") as f:
            f.write(rst)


def _package_namespace(package_name: str) -> str:
    """Returns the package name used.

    Args:
        package_name: Can be either "langchain" or "core" or "experimental".

    Returns:
        modified package_name: Can be either "langchain" or "langchain_{package_name}"
    """
    if package_name == "langchain":
        return "langchain"
    if package_name == "standard-tests":
        return "langchain_tests"
    return f"langchain_{package_name.replace('-', '_')}"


def _package_dir(package_name: str = "langchain") -> Path:
    """Return the path to the directory containing the documentation."""
    if package_name in (
        "langchain",
        "experimental",
        "community",
        "core",
        "cli",
        "text-splitters",
        "standard-tests",
    ):
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
    return HERE / f"{package_name.replace('-', '_')}"


def _build_index(dirs: List[str]) -> None:
    custom_names = {
        "aws": "AWS",
        "ai21": "AI21",
        "ibm": "IBM",
    }
    ordered = ["core", "langchain", "text-splitters", "community", "experimental"]
    main_ = [dir_ for dir_ in ordered if dir_ in dirs]
    integrations = sorted(dir_ for dir_ in dirs if dir_ not in main_)
    doc = """# LangChain Python API Reference

Welcome to the LangChain Python API reference. This is a reference for all 
`langchain-x` packages. 

For user guides see [https://python.langchain.com](https://python.langchain.com).

For the legacy API reference hosted on ReadTheDocs see [https://api.python.langchain.com/](https://api.python.langchain.com/).
"""

    if main_:
        main_headers = [
            " ".join(custom_names.get(x, x.title()) for x in dir_.split("-"))
            for dir_ in main_
        ]
        main_tree = "\n".join(
            f"{header_name}<{dir_.replace('-', '_')}/index>"
            for header_name, dir_ in zip(main_headers, main_)
        )
        main_grid = "\n".join(
            f'- header: "**{header_name}**"\n  content: "{_package_namespace(dir_).replace("_", "-")}: {_get_package_version(_package_dir(dir_))}"\n  link: {dir_.replace("-", "_")}/index.html'
            for header_name, dir_ in zip(main_headers, main_)
        )
        doc += f"""## Base packages

```{{gallery-grid}}
:grid-columns: "1 2 2 3"

{main_grid}
```

```{{toctree}}
:maxdepth: 2
:hidden:
:caption: Base packages

{main_tree}
```
"""
    if integrations:
        integration_headers = [
            " ".join(
                custom_names.get(x, x.title().replace("ai", "AI").replace("db", "DB"))
                for x in dir_.split("-")
            )
            for dir_ in integrations
        ]
        integration_tree = "\n".join(
            f"{header_name}<{dir_.replace('-', '_')}/index>"
            for header_name, dir_ in zip(integration_headers, integrations)
        )

        integration_grid = ""
        integrations_to_show = [
            "openai",
            "anthropic",
            "google-vertexai",
            "aws",
            "huggingface",
            "mistralai",
        ]
        for header_name, dir_ in sorted(
            zip(integration_headers, integrations),
            key=lambda h_d: (
                integrations_to_show.index(h_d[1])
                if h_d[1] in integrations_to_show
                else len(integrations_to_show)
            ),
        )[: len(integrations_to_show)]:
            integration_grid += f'\n- header: "**{header_name}**"\n  content: {_package_namespace(dir_).replace("_", "-")} {_get_package_version(_package_dir(dir_))}\n  link: {dir_.replace("-", "_")}/index.html'
        doc += f"""## Integrations

```{{gallery-grid}}
:grid-columns: "1 2 2 3"

{integration_grid}
```

See the full list of integrations in the Section Navigation.

```{{toctree}}
:maxdepth: 2
:hidden:
:caption: Integrations

{integration_tree}
```
"""
    with open(HERE / "reference.md", "w") as f:
        f.write(doc)

    dummy_index = """\
# API reference

```{toctree}
:maxdepth: 3
:hidden:

Reference<reference>
```
"""
    with open(HERE / "index.md", "w") as f:
        f.write(dummy_index)


def main(dirs: Optional[list] = None) -> None:
    """Generate the api_reference.rst file for each package."""
    print("Starting to build API reference files.")
    if not dirs:
        dirs = [
            dir_
            for dir_ in os.listdir(ROOT_DIR / "libs")
            if dir_ not in ("cli", "partners", "packages.yml")
        ]
        dirs += [
            dir_
            for dir_ in os.listdir(ROOT_DIR / "libs" / "partners")
            if os.path.isdir(ROOT_DIR / "libs" / "partners" / dir_)
            and "pyproject.toml" in os.listdir(ROOT_DIR / "libs" / "partners" / dir_)
        ]
    for dir_ in dirs:
        # Skip any hidden directories
        # Some of these could be present by mistake in the code base
        # e.g., .pytest_cache from running tests from the wrong location.
        if dir_.startswith("."):
            print("Skipping dir:", dir_)
            continue
        else:
            print("Building package:", dir_)
            _build_rst_file(package_name=dir_)

    _build_index(dirs)
    print("API reference files built.")


if __name__ == "__main__":
    dirs = sys.argv[1:] or None
    main(dirs=dirs)
