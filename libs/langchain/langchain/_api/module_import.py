import importlib
import os
import pathlib
import warnings
from typing import Any, Callable, Dict, Optional

from langchain_core._api import LangChainDeprecationWarning

from langchain.utils.interactive_env import is_interactive_env

ALLOWED_TOP_LEVEL_PKGS = {
    "langchain_community",
    "langchain_core",
    "langchain",
}


HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent.parent


def _get_current_module(path: str) -> str:
    """Convert a path to a module name."""
    path_as_pathlib = pathlib.Path(os.path.abspath(path))
    relative_path = path_as_pathlib.relative_to(ROOT).with_suffix("")
    posix_path = relative_path.as_posix()
    norm_path = os.path.normpath(str(posix_path))
    fully_qualified_module = norm_path.replace("/", ".")
    # Strip off __init__ if present
    if fully_qualified_module.endswith(".__init__"):
        return fully_qualified_module[:-9]
    return fully_qualified_module


def create_importer(
    here: str,
    *,
    module_lookup: Optional[Dict[str, str]] = None,
    fallback_module: Optional[str] = None,
) -> Callable[[str], Any]:
    """Create a function that helps retrieve objects from their new locations.

    The goal of this function is to help users transition from deprecated
    imports to new imports.

    This function will raise warnings when the old imports are used and
    suggest the new imports.

    This function should ideally only be used with deprecated imports not with
    existing imports that are valid, as in addition to raising deprecation warnings
    the dynamic imports can create other issues for developers (e.g.,
    loss of type information, IDE support for going to definition etc).

    Args:
        here: path of the current file. Use __file__
        module_lookup: maps name of object to the module where it is defined.
            e.g.,
            {
                "MyDocumentLoader": (
                    "langchain_community.document_loaders.my_document_loader"
                )
            }
        fallback_module: module to import from if the object is not found in
            module_lookup or if module_lookup is not provided.

    Returns:
        A function that imports objects from the specified modules.
    """
    current_module = _get_current_module(here)

    def import_by_name(name: str) -> Any:
        """Import stores from langchain_community."""
        # If not in interactive env, raise warning.
        if module_lookup and name in module_lookup:
            new_module = module_lookup[name]
            if new_module.split(".")[0] not in ALLOWED_TOP_LEVEL_PKGS:
                raise AssertionError(
                    f"Importing from {new_module} is not allowed. "
                    f"Allowed top-level packages are: {ALLOWED_TOP_LEVEL_PKGS}"
                )

            try:
                module = importlib.import_module(new_module)
            except ModuleNotFoundError as e:
                if new_module.startswith("langchain_community"):
                    raise ModuleNotFoundError(
                        f"Module {new_module} not found. "
                        "Please install langchain-community to access this module. "
                        "You can install it using `pip install -U langchain-community`"
                    ) from e
                raise

            try:
                result = getattr(module, name)
                if not is_interactive_env():
                    warnings.warn(
                        f"Importing {name} from {current_module} is deprecated. "
                        "Please replace the import with the following:\n"
                        f"from {new_module} import {name}",
                        category=LangChainDeprecationWarning,
                    )
                return result
            except Exception as e:
                raise AttributeError(
                    f"module {new_module} has no attribute {name}"
                ) from e

        if fallback_module:
            try:
                module = importlib.import_module(fallback_module)
                result = getattr(module, name)
                if not is_interactive_env():
                    warnings.warn(
                        f"Importing {name} from {current_module} is deprecated. "
                        "Please replace the import with the following:\n"
                        f"from {fallback_module} import {name}",
                        category=LangChainDeprecationWarning,
                    )
                return result

            except Exception as e:
                raise AttributeError(
                    f"module {fallback_module} has no attribute {name}"
                ) from e

        raise AttributeError(f"module {current_module} has no attribute {name}")

    return import_by_name
