import ast
import glob
import importlib
from pathlib import Path

import pytest

COMMUNITY_ROOT = Path(__file__).parent.parent.parent / "langchain_community"
ALL_COMMUNITY_GLOB = COMMUNITY_ROOT.as_posix() + "/**/*.py"
HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def test_importable_all() -> None:
    for path in glob.glob(ALL_COMMUNITY_GLOB):
        # Relative to community root
        relative_path = Path(path).relative_to(COMMUNITY_ROOT)
        str_path = str(relative_path)
        if str_path.endswith("__init__.py"):
            module_name = str(relative_path.parent).replace("/", ".")
        else:
            module_name = str(relative_path.with_suffix("")).replace("/", ".")

        try:
            module = importlib.import_module("langchain_community." + module_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Could not import `{module_name}`. Defined in path: {path}"
            ) from e
        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            getattr(module, cls_)


def test_glob_correct() -> None:
    """Verify that the glob pattern is correct."""
    paths = list(glob.glob(ALL_COMMUNITY_GLOB))
    # Get paths relative to community root
    paths_ = [Path(path).relative_to(COMMUNITY_ROOT) for path in paths]
    # Assert there's a callback paths
    assert Path("callbacks/__init__.py") in paths_


def _check_correct_or_not_defined__all__(code: str) -> bool:
    """Return True if __all__ is correctly defined or not defined at all."""
    # Parse the code into an AST
    tree = ast.parse(code)

    all_good = True

    # Iterate through the body of the AST to find assignments
    for node in tree.body:
        # Check if the node is an assignment
        if isinstance(node, ast.Assign):
            # Check if the target of the assignment is '__all__'
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    # Check if the value assigned is a list
                    if isinstance(node.value, ast.List):
                        # Verify all elements in the list are string literals
                        if all(isinstance(el, ast.Str) for el in node.value.elts):
                            pass
                        else:
                            all_good = False
                    else:
                        all_good = False
    return all_good


@pytest.mark.xfail
def test_no_dynamic__all__() -> None:
    """Verify that __all__ is not computed at runtime.

    Computing __all__ dynamically can confuse static typing tools like pyright.

    __all__ should always be listed as an explicit list of string literals.
    """
    bad_definitions = []
    for path in glob.glob(ALL_COMMUNITY_GLOB):
        if not path.endswith("__init__.py"):
            continue

        with open(path, "r") as file:
            code = file.read()

        if _check_correct_or_not_defined__all__(code) is False:
            bad_definitions.append(path)

    if bad_definitions:
        raise AssertionError(
            f"__all__ is not correctly defined in the "
            f"following files: {sorted(bad_definitions)}"
        )
