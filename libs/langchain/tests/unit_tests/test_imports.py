import ast
import importlib
import warnings
from pathlib import Path
from typing import Any, Dict, Optional

# Attempt to recursively import all modules in langchain
PKG_ROOT = Path(__file__).parent.parent.parent

COMMUNITY_NOT_INSTALLED = importlib.util.find_spec("langchain_community") is None


def test_import_all() -> None:
    """Generate the public API for this package."""
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", category=UserWarning)
        library_code = PKG_ROOT / "langchain"
        for path in library_code.rglob("*.py"):
            # Calculate the relative path to the module
            module_name = (
                path.relative_to(PKG_ROOT).with_suffix("").as_posix().replace("/", ".")
            )
            if module_name.endswith("__init__"):
                # Without init
                module_name = module_name.rsplit(".", 1)[0]

            mod = importlib.import_module(module_name)

            all = getattr(mod, "__all__", [])

            for name in all:
                # Attempt to import the name from the module
                try:
                    obj = getattr(mod, name)
                    assert obj is not None
                except ModuleNotFoundError as e:
                    # If the module is not installed, we suppress the error
                    if (
                        "Module langchain_community" in str(e)
                        and COMMUNITY_NOT_INSTALLED
                    ):
                        pass
                except Exception as e:
                    raise AssertionError(
                        f"Could not import {module_name}.{name}"
                    ) from e


def test_import_all_using_dir() -> None:
    """Generate the public API for this package."""
    library_code = PKG_ROOT / "langchain"
    for path in library_code.rglob("*.py"):
        # Calculate the relative path to the module
        module_name = (
            path.relative_to(PKG_ROOT).with_suffix("").as_posix().replace("/", ".")
        )
        if module_name.endswith("__init__"):
            # Without init
            module_name = module_name.rsplit(".", 1)[0]

        if module_name.startswith("langchain_community.") and COMMUNITY_NOT_INSTALLED:
            continue

        try:
            mod = importlib.import_module(module_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(f"Could not import {module_name}") from e
        all = dir(mod)

        for name in all:
            if name.strip().startswith("_"):
                continue
            # Attempt to import the name from the module
            getattr(mod, name)


def test_no_more_changes_to_proxy_community() -> None:
    """This test is meant to catch any changes to the proxy community module.

    Imports from langchain to community are officially DEPRECATED. Contributors
    should not be adding new imports from langchain to community. This test
    is meant to catch any new changes to the proxy community module.
    """
    library_code = PKG_ROOT / "langchain"
    hash_ = 0
    for path in library_code.rglob("*.py"):
        # Calculate the relative path to the module
        if not str(path).endswith("__init__.py"):
            continue

        deprecated_lookup = extract_deprecated_lookup(str(path))
        if deprecated_lookup is None:
            continue

        # This uses a very simple hash, so it's not foolproof, but it should catch
        # most cases.
        hash_ += len(str(sorted(deprecated_lookup.items())))

    evil_magic_number = 38697

    assert hash_ == evil_magic_number, (
        "If you're triggering this test, you're likely adding a new import "
        "to the langchain package that is importing something from "
        "langchain_community. This test is meant to catch such such imports "
        "as they are officially DEPRECATED. Please do not add any new imports "
        "from langchain_community to the langchain package. "
    )


def extract_deprecated_lookup(file_path: str) -> Optional[Dict[str, Any]]:
    """Detect and extracts the value of a dictionary named DEPRECATED_LOOKUP

    This variable is located in the global namespace of a Python file.

    Args:
        file_path (str): The path to the Python file.

    Returns:
        dict or None: The value of DEPRECATED_LOOKUP if it exists, None otherwise.
    """
    with open(file_path, "r") as file:
        tree = ast.parse(file.read(), filename=file_path)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "DEPRECATED_LOOKUP":
                    if isinstance(node.value, ast.Dict):
                        return _dict_from_ast(node.value)
    return None


def _dict_from_ast(node: ast.Dict) -> Dict[str, str]:
    """Convert an AST dict node to a Python dictionary, assuming str to str format.

    Args:
        node (ast.Dict): The AST node representing a dictionary.

    Returns:
        dict: The corresponding Python dictionary.
    """
    result: Dict[str, str] = {}
    for key, value in zip(node.keys, node.values):
        py_key = _literal_eval_str(key)  # type: ignore
        py_value = _literal_eval_str(value)
        result[py_key] = py_value
    return result


def _literal_eval_str(node: ast.AST) -> str:
    """Evaluate an AST literal node to its corresponding string value.

    Args:
        node (ast.AST): The AST node representing a literal value.

    Returns:
        str: The corresponding string value.
    """
    if isinstance(node, ast.Constant):  # Python 3.8+
        if isinstance(node.value, str):
            return node.value
    raise AssertionError(
        f"Invalid DEPRECATED_LOOKUP format: expected str, got {type(node).__name__}"
    )
