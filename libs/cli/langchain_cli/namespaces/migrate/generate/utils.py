import ast
import inspect
import os
import pathlib
from pathlib import Path
from typing import Any, List, Optional, Tuple, Type

HERE = Path(__file__).parent
# Should bring us to [root]/src
PKGS_ROOT = HERE.parent.parent.parent.parent.parent

LANGCHAIN_PKG = PKGS_ROOT / "langchain"
COMMUNITY_PKG = PKGS_ROOT / "community"
PARTNER_PKGS = PKGS_ROOT / "partners"


class ImportExtractor(ast.NodeVisitor):
    def __init__(self, *, from_package: Optional[str] = None) -> None:
        """Extract all imports from the given code, optionally filtering by package."""
        self.imports = []
        self.package = from_package

    def visit_ImportFrom(self, node):
        if node.module and (
            self.package is None or str(node.module).startswith(self.package)
        ):
            for alias in node.names:
                self.imports.append((node.module, alias.name))
        self.generic_visit(node)


def _get_class_names(code: str) -> List[str]:
    """Extract class names from a code string."""
    # Parse the content of the file into an AST
    tree = ast.parse(code)

    # Initialize a list to hold all class names
    class_names = []

    # Define a node visitor class to collect class names
    class ClassVisitor(ast.NodeVisitor):
        def visit_ClassDef(self, node):
            class_names.append(node.name)
            self.generic_visit(node)

    # Create an instance of the visitor and visit the AST
    visitor = ClassVisitor()
    visitor.visit(tree)
    return class_names


def is_subclass(class_obj: Any, classes_: List[Type]) -> bool:
    """Check if the given class object is a subclass of any class in list classes."""
    return any(
        issubclass(class_obj, kls)
        for kls in classes_
        if inspect.isclass(class_obj) and inspect.isclass(kls)
    )


def find_subclasses_in_module(module, classes_: List[Type]) -> List[str]:
    """Find all classes in the module that inherit from one of the classes."""
    subclasses = []
    # Iterate over all attributes of the module that are classes
    for name, obj in inspect.getmembers(module, inspect.isclass):
        if is_subclass(obj, classes_):
            subclasses.append(obj.__name__)
    return subclasses


def _get_all_classnames_from_file(file: str, pkg: str) -> List[Tuple[str, str]]:
    """Extract all class names from a file."""
    with open(file, encoding="utf-8") as f:
        code = f.read()
    module_name = _get_current_module(file, pkg)
    class_names = _get_class_names(code)

    return [(module_name, class_name) for class_name in class_names]


def identify_all_imports_in_file(
    file: str, *, from_package: Optional[str] = None
) -> List[Tuple[str, str]]:
    """Let's also identify all the imports in the given file."""
    with open(file, encoding="utf-8") as f:
        code = f.read()
    return find_imports_from_package(code, from_package=from_package)


def identify_pkg_source(pkg_root: str) -> pathlib.Path:
    """Identify the source of the package.

    Args:
        pkg_root: the root of the package. This contains source + tests, and other
            things like pyproject.toml, lock files etc

    Returns:
        Returns the path to the source code for the package.
    """
    dirs = [d for d in Path(pkg_root).iterdir() if d.is_dir()]
    matching_dirs = [d for d in dirs if d.name.startswith("langchain_")]
    assert len(matching_dirs) == 1, "There should be only one langchain package."
    return matching_dirs[0]


def list_classes_by_package(pkg_root: str) -> List[Tuple[str, str]]:
    """List all classes in a package."""
    module_classes = []
    pkg_source = identify_pkg_source(pkg_root)
    files = list(pkg_source.rglob("*.py"))

    for file in files:
        rel_path = os.path.relpath(file, pkg_root)
        if rel_path.startswith("tests"):
            continue
        module_classes.extend(_get_all_classnames_from_file(file, pkg_root))
    return module_classes


def list_init_imports_by_package(pkg_root: str) -> List[Tuple[str, str]]:
    """List all the things that are being imported in a package by module."""
    imports = []
    pkg_source = identify_pkg_source(pkg_root)
    # Scan all the files in the package
    files = list(Path(pkg_source).rglob("*.py"))

    for file in files:
        if not file.name == "__init__.py":
            continue
        import_in_file = identify_all_imports_in_file(str(file))
        module_name = _get_current_module(file, pkg_root)
        imports.extend([(module_name, item) for _, item in import_in_file])
    return imports


def find_imports_from_package(
    code: str, *, from_package: Optional[str] = None
) -> List[Tuple[str, str]]:
    # Parse the code into an AST
    tree = ast.parse(code)
    # Create an instance of the visitor
    extractor = ImportExtractor(from_package=from_package)
    # Use the visitor to update the imports list
    extractor.visit(tree)
    return extractor.imports


def _get_current_module(path: str, pkg_root: str) -> str:
    """Convert a path to a module name."""
    path_as_pathlib = pathlib.Path(os.path.abspath(path))
    relative_path = path_as_pathlib.relative_to(pkg_root).with_suffix("")
    posix_path = relative_path.as_posix()
    norm_path = os.path.normpath(str(posix_path))
    fully_qualified_module = norm_path.replace("/", ".")
    # Strip __init__ if present
    if fully_qualified_module.endswith(".__init__"):
        return fully_qualified_module[:-9]
    return fully_qualified_module
