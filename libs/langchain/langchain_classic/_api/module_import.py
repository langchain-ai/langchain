import importlib
from collections.abc import Callable
from typing import Any

from langchain_core._api import internal, warn_deprecated

from langchain_classic._api.interactive_env import is_interactive_env

ALLOWED_TOP_LEVEL_PKGS = {
    "langchain_community",
    "langchain_core",
    "langchain_classic",
}


def create_importer(
    package: str,
    *,
    module_lookup: dict[str, str] | None = None,
    deprecated_lookups: dict[str, str] | None = None,
    fallback_module: str | None = None,
) -> Callable[[str], Any]:
    """Create a function that helps retrieve objects from their new locations.

    The goal of this function is to help users transition from deprecated
    imports to new imports.

    The function will raise deprecation warning on loops using
    `deprecated_lookups` or `fallback_module`.

    Module lookups will import without deprecation warnings (used to speed
    up imports from large namespaces like llms or chat models).

    This function should ideally only be used with deprecated imports not with
    existing imports that are valid, as in addition to raising deprecation warnings
    the dynamic imports can create other issues for developers (e.g.,
    loss of type information, IDE support for going to definition etc).

    Args:
        package: Current package. Use `__package__`
        module_lookup: Maps name of object to the module where it is defined.
            e.g.,
            ```json
            {
                "MyDocumentLoader": (
                    "langchain_community.document_loaders.my_document_loader"
                )
            }
            ```
        deprecated_lookups: Same as module look up, but will raise
            deprecation warnings.
        fallback_module: Module to import from if the object is not found in
            `module_lookup` or if `module_lookup` is not provided.

    Returns:
        A function that imports objects from the specified modules.
    """
    all_module_lookup = {**(deprecated_lookups or {}), **(module_lookup or {})}

    def import_by_name(name: str) -> Any:
        """Import stores from `langchain_community`."""
        # If not in interactive env, raise warning.
        if all_module_lookup and name in all_module_lookup:
            new_module = all_module_lookup[name]
            if new_module.split(".")[0] not in ALLOWED_TOP_LEVEL_PKGS:
                msg = (
                    f"Importing from {new_module} is not allowed. "
                    f"Allowed top-level packages are: {ALLOWED_TOP_LEVEL_PKGS}"
                )
                raise AssertionError(msg)

            try:
                module = importlib.import_module(new_module)
            except ModuleNotFoundError as e:
                if new_module.startswith("langchain_community"):
                    msg = (
                        f"Module {new_module} not found. "
                        "Please install langchain-community to access this module. "
                        "You can install it using `pip install -U langchain-community`"
                    )
                    raise ModuleNotFoundError(msg) from e
                raise

            try:
                result = getattr(module, name)
                if (
                    not is_interactive_env()
                    and deprecated_lookups
                    and name in deprecated_lookups
                    # Depth 3:
                    # -> internal.py
                    # |-> module_import.py
                    #  |-> Module in langchain that uses this function
                    #   |-> [calling code] whose frame we want to inspect.
                    and not internal.is_caller_internal(depth=3)
                ):
                    warn_deprecated(
                        since="0.1",
                        pending=False,
                        removal="1.0",
                        message=(
                            f"Importing {name} from {package} is deprecated. "
                            f"Please replace deprecated imports:\n\n"
                            f">> from {package} import {name}\n\n"
                            "with new imports of:\n\n"
                            f">> from {new_module} import {name}\n"
                            "You can use the langchain cli to **automatically** "
                            "upgrade many imports. Please see documentation here "
                            "<https://python.langchain.com/docs/versions/v0_2/>"
                        ),
                    )
            except Exception as e:
                msg = f"module {new_module} has no attribute {name}"
                raise AttributeError(msg) from e

            return result

        if fallback_module:
            try:
                module = importlib.import_module(fallback_module)
                result = getattr(module, name)
                if (
                    not is_interactive_env()
                    # Depth 3:
                    # internal.py
                    # |-> module_import.py
                    #  |->Module in langchain that uses this function
                    #   |-> [calling code] whose frame we want to inspect.
                    and not internal.is_caller_internal(depth=3)
                ):
                    warn_deprecated(
                        since="0.1",
                        pending=False,
                        removal="1.0",
                        message=(
                            f"Importing {name} from {package} is deprecated. "
                            f"Please replace deprecated imports:\n\n"
                            f">> from {package} import {name}\n\n"
                            "with new imports of:\n\n"
                            f">> from {fallback_module} import {name}\n"
                            "You can use the langchain cli to **automatically** "
                            "upgrade many imports. Please see documentation here "
                            "<https://python.langchain.com/docs/versions/v0_2/>"
                        ),
                    )

            except Exception as e:
                msg = f"module {fallback_module} has no attribute {name}"
                raise AttributeError(msg) from e

            return result

        msg = f"module {package} has no attribute {name}"
        raise AttributeError(msg)

    return import_by_name
