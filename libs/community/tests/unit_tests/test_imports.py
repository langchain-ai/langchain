import ast
import glob
import importlib
from pathlib import Path

COMMUNITY_ROOT = Path(__file__).parent.parent.parent / "langchain_community"
ALL_COMMUNITY_GLOB = COMMUNITY_ROOT.as_posix() + "/**/*.py"
HERE = Path(__file__).parent
ROOT = HERE.parent.parent


def test_importable_all() -> None:
    for path in glob.glob(ALL_COMMUNITY_GLOB):
        relative_path = Path(path).parts[-1]
        if relative_path.endswith(".typed"):
            continue
        module_name = relative_path.split(".")[0]

        module = importlib.import_module("langchain_community." + module_name)
        all_ = getattr(module, "__all__", [])
        for cls_ in all_:
            getattr(module, cls_)


def test_glob_correct() -> None:
    """Verify that the glob pattern is correct."""
    paths = list(glob.glob(ALL_COMMUNITY_GLOB))
    # Get paths relative to community root
    paths = [Path(path).relative_to(COMMUNITY_ROOT) for path in paths]
    # Assert there's a callback paths
    assert Path("callbacks/__init__.py") in paths


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


def test_no_dynamic__all__() -> None:
    """Verify that __all__ is not computed at runtime.

    Computing __all__ dynamically can confuse static typing tools like pyright.

    __all__ should always be listed as an explicit list of string literals.
    """
    for path in glob.glob(ALL_COMMUNITY_GLOB):
        relative_path = Path(path).parts[-1]
        if relative_path.endswith(".typed"):
            continue

        if not path.endswith("__init__.py"):
            continue

        with open(path, "r") as file:
            code = file.read()

        assert _check_correct_or_not_defined__all__(
            code
        ), f"__all__ is not correctly defined in {path}"


def test_packages() -> None:
    """
    Inspects a Python file to validate consistency between TYPE_CHECKING imports,
    the _module_lookup dictionary, and the __all__ list.

    Specifically, this function performs the following checks:
    - Ensures that all imports listed under a TYPE_CHECKING conditional
              match the keys in the _module_lookup dictionary.
    - Checks that __all__ is defined as a list of string literals and
              matches the keys of the _module_lookup dictionary.
    - Validates that __all__ is not computed at runtime but defined
                directly with the correct module attributes.
    """
    # Get all top level packages in PARENT and read their __init__ file
    package_root = ROOT
    # Traverse all child directories and open __init__.py files in them if found
    for path in package_root.glob("langchain_community/*"):
        if path.is_dir():
            init_file = path / "__init__.py"
            if not init_file.exists():
                continue
            # Read and parse the Python file
            file_path = str(init_file)
            with open(file_path, "r") as file:
                tree = ast.parse(file.read(), filename=file_path)

            # Initialize containers for TYPE_CHECKING imports and _module_lookup
            type_checking_imports = []
            module_lookup = []

            # Traverse the AST to find relevant nodes
            for node in ast.walk(tree):
                # Check for TYPE_CHECKING imports
                if (
                    isinstance(node, ast.If)
                    and isinstance(node.test, ast.Name)
                    and node.test.id == "TYPE_CHECKING"
                ):
                    for body_item in node.body:
                        if isinstance(body_item, ast.ImportFrom):
                            for alias in body_item.names:
                                type_checking_imports.append(alias.name)

                # Check for _module_lookup definition
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if (
                            isinstance(target, ast.Name)
                            and target.id == "_module_lookup"
                        ):
                            if isinstance(node.value, ast.Dict):
                                module_lookup = [key.s for key in node.value.keys]

            # Assert that TYPE_CHECKING imports match _module_lookup
            assert set(type_checking_imports) == set(
                module_lookup
            ), "TYPE_CHECKING imports do not match _module_keys in {}".format(file_path)

            # Check for __all__ definition directly in the file
            all_defined_correctly = False
            all_values = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "__all__":
                            if isinstance(node.value, ast.List) and all(
                                isinstance(elem, ast.Str) for elem in node.value.elts
                            ):
                                all_defined_correctly = True
                                all_values = [elem.s for elem in node.value.elts]

            # Assert that __all__ is correctly defined as a list of string literals
            # and matches the module lookup
            assert (
                all_defined_correctly
            ), "__all__ must be a list of string literals in {}".format(file_path)
            assert (
                all_values == module_lookup
            ), "__all__ does not match module_lookup keys in {}".format(file_path)
