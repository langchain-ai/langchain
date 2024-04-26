import ast
import inspect
import os
import pathlib
from pathlib import Path
from typing import Any, List, Tuple, Type

HERE = Path(__file__).parent
# Should bring us to [root]/src
PKGS_ROOT = HERE.parent.parent.parent.parent.parent

LANGCHAIN_PKG = PKGS_ROOT / "langchain"
COMMUNITY_PKG = PKGS_ROOT / "community"
PARTNER_PKGS = PKGS_ROOT / "partners"


class ImportExtractor(ast.NodeVisitor):
    def __init__(self, *, from_package: str) -> None:
        """Extract all imports from the given package."""
        self.imports = []
        self.package = from_package

    def visit_ImportFrom(self, node):
        if node.module and str(node.module).startswith(self.package):
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


def list_classes_by_package(pkg_root: str) -> List[Tuple[str, str]]:
    """List all classes in a package."""
    module_classes = []
    files = list(Path(pkg_root).rglob("*.py"))

    for file in files:
        rel_path = os.path.relpath(file, pkg_root)
        if rel_path.startswith("tests"):
            continue
        module_classes.extend(_get_all_classnames_from_file(file, pkg_root))
    return module_classes


def find_imports_from_package(code: str, *, from_package: str) -> List[Tuple[str, str]]:
    # Parse the code into an AST
    tree = ast.parse(code)
    # Create an instance of the visitor
    extractor = ImportExtractor(from_package="langchain_community")
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
