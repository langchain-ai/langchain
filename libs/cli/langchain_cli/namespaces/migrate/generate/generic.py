"""Generate migrations from langchain to langchain-community or core packages."""

import importlib
import inspect
import pkgutil
from typing import List, Tuple


def generate_raw_migrations(
    from_package: str, to_package: str, filter_by_all: bool = False
) -> List[Tuple[str, str]]:
    """Scan the `langchain` package and generate migrations for all modules."""
    package = importlib.import_module(from_package)

    items = []
    for importer, modname, ispkg in pkgutil.walk_packages(
        package.__path__, package.__name__ + "."
    ):
        try:
            module = importlib.import_module(modname)
        except ModuleNotFoundError:
            continue

        # Check if the module is an __init__ file and evaluate __all__
        try:
            has_all = hasattr(module, "__all__")
        except ImportError:
            has_all = False

        if has_all:
            all_objects = module.__all__
            for name in all_objects:
                # Attempt to fetch each object declared in __all__
                try:
                    obj = getattr(module, name, None)
                except ImportError:
                    continue
                if obj and (inspect.isclass(obj) or inspect.isfunction(obj)):
                    if obj.__module__.startswith(to_package):
                        items.append(
                            (f"{modname}.{name}", f"{obj.__module__}.{obj.__name__}")
                        )

        if not filter_by_all:
            # Iterate over all members of the module
            for name, obj in inspect.getmembers(module):
                # Check if it's a class or function
                if inspect.isclass(obj) or inspect.isfunction(obj):
                    # Check if the module name of the obj starts with
                    # 'langchain_community'
                    if obj.__module__.startswith(to_package):
                        items.append(
                            (f"{modname}.{name}", f"{obj.__module__}.{obj.__name__}")
                        )

    return items


def generate_top_level_imports(pkg: str) -> List[Tuple[str, str]]:
    """This code will look at all the top level modules in langchain_community.

    It'll attempt to import everything from each __init__ file

    for example,

    langchain_community/
        chat_models/
            __init__.py # <-- import everything from here
        llm/
            __init__.py # <-- import everything from here


    It'll collect all the imports, import the classes / functions it can find
    there. It'll return a list of 2-tuples

    Each tuple will contain the fully qualified path of the class / function to where
    its logic is defined
    (e.g., langchain_community.chat_models.xyz_implementation.ver2.XYZ)
    and the second tuple will contain the path
    to importing it from the top level namespaces
    (e.g., langchain_community.chat_models.XYZ)
    """
    package = importlib.import_module(pkg)

    items = []

    # Function to handle importing from modules
    def handle_module(module, module_name):
        if hasattr(module, "__all__"):
            all_objects = getattr(module, "__all__")
            for name in all_objects:
                # Attempt to fetch each object declared in __all__
                obj = getattr(module, name, None)
                if obj and (inspect.isclass(obj) or inspect.isfunction(obj)):
                    # Capture the fully qualified name of the object
                    original_module = obj.__module__
                    original_name = obj.__name__
                    # Form the new import path from the top-level namespace
                    top_level_import = f"{module_name}.{name}"
                    # Append the tuple with original and top-level paths
                    items.append(
                        (f"{original_module}.{original_name}", top_level_import)
                    )

    # Handle the package itself (root level)
    handle_module(package, pkg)

    # Only iterate through top-level modules/packages
    for finder, modname, ispkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        if ispkg:
            try:
                module = importlib.import_module(modname)
                handle_module(module, modname)
            except ModuleNotFoundError:
                continue

    return items


def generate_simplified_migrations(
    from_package: str, to_package: str, filter_by_all: bool = True
) -> List[Tuple[str, str]]:
    """Get all the raw migrations, then simplify them if possible."""
    raw_migrations = generate_raw_migrations(
        from_package, to_package, filter_by_all=filter_by_all
    )
    top_level_simplifications = generate_top_level_imports(to_package)
    top_level_dict = {full: top_level for full, top_level in top_level_simplifications}
    simple_migrations = []
    for migration in raw_migrations:
        original, new = migration
        replacement = top_level_dict.get(new, new)
        simple_migrations.append((original, replacement))

    # Now let's deduplicate the list based on the original path (which is
    # the 1st element of the tuple)
    deduped_migrations = []
    seen = set()
    for migration in simple_migrations:
        original = migration[0]
        if original not in seen:
            deduped_migrations.append(migration)
            seen.add(original)

    return deduped_migrations
