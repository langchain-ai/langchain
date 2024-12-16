import ast
import glob
import importlib
from pathlib import Path
from typing import List, Tuple

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


def _extract_type_checking_imports(code: str) -> List[Tuple[str, str]]:
    """Extract all TYPE CHECKING imports that import from langchain_community."""
    imports: List[Tuple[str, str]] = []

    tree = ast.parse(code)

    class TypeCheckingVisitor(ast.NodeVisitor):
        def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
            if node.module:
                for alias in node.names:
                    imports.append((node.module, alias.name))

    class GlobalScopeVisitor(ast.NodeVisitor):
        def visit_If(self, node: ast.If) -> None:
            if (
                isinstance(node.test, ast.Name)
                and node.test.id == "TYPE_CHECKING"
                and isinstance(node.test.ctx, ast.Load)
            ):
                TypeCheckingVisitor().visit(node)
            self.generic_visit(node)

    GlobalScopeVisitor().visit(tree)
    return imports


def test_init_files_properly_defined() -> None:
    """This is part of a set of tests that verify that init files are properly

    defined if they're using dynamic imports.
    """
    # Please never ever add more modules to this list.
    # Do feel free to fix the underlying issues and remove exceptions
    # from the list.
    excepted_modules = {"llms"}  # NEVER ADD MORE MODULES TO THIS LIST
    for path in glob.glob(ALL_COMMUNITY_GLOB):
        # Relative to community root
        relative_path = Path(path).relative_to(COMMUNITY_ROOT)
        str_path = str(relative_path)

        if not str_path.endswith("__init__.py"):
            continue

        module_name = str(relative_path.parent).replace("/", ".")

        if module_name in excepted_modules:
            continue

        code = Path(path).read_text()

        # Check for dynamic __getattr__ definition in the __init__ file
        if "__getattr__" not in code:
            continue

        try:
            module = importlib.import_module("langchain_community." + module_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                f"Could not import `{module_name}`. Defined in path: {path}"
            ) from e

        if not hasattr(module, "__all__"):
            raise AssertionError(
                f"__all__ not defined in {module_name}. This is required "
                f"if __getattr__ is defined."
            )

        imports = _extract_type_checking_imports(code)

        # Get the names of all the TYPE CHECKING imports
        names = [name for _, name in imports]

        missing_imports = set(module.__all__) - set(names)

        assert (
            not missing_imports
        ), f"Missing imports: {missing_imports} in file path: {path}"
