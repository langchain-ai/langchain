import ast
import os
import sys
from pathlib import Path


class ImportTransformer(ast.NodeTransformer):
    def __init__(self, public_items, module_name):
        self.public_items = public_items
        self.module_name = module_name

    def visit_Module(self, node):
        imports = [
            ast.ImportFrom(
                module=self.module_name,
                names=[ast.alias(name=item, asname=None)],
                level=0,
            )
            for item in self.public_items
        ]
        all_assignment = ast.Assign(
            targets=[ast.Name(id="__all__", ctx=ast.Store())],
            value=ast.List(
                elts=[ast.Str(s=item) for item in self.public_items], ctx=ast.Load()
            ),
        )
        node.body = imports + [all_assignment]
        return node


def find_public_classes_and_methods(file_path):
    with open(file_path, "r") as file:
        node = ast.parse(file.read(), filename=file_path)

    public_items = []
    for item in node.body:
        if isinstance(item, ast.ClassDef) or isinstance(item, ast.FunctionDef):
            public_items.append(item.name)
        if (
            isinstance(item, ast.Assign)
            and hasattr(item.targets[0], "id")
            and item.targets[0].id not in ("__all__", "logger")
        ):
            public_items.append(item.targets[0].id)

    return public_items or None


def process_file(file_path, module_name):
    public_items = find_public_classes_and_methods(file_path)
    if public_items is None:
        return

    with open(file_path, "r") as file:
        contents = file.read()
        tree = ast.parse(contents, filename=file_path)

    tree = ImportTransformer(public_items, module_name).visit(tree)
    tree = ast.fix_missing_locations(tree)

    with open(file_path, "w") as file:
        file.write(ast.unparse(tree))


def process_directory(directory_path, base_module_name):
    if Path(directory_path).is_file():
        process_file(directory_path, base_module_name)
    else:
        for root, dirs, files in os.walk(directory_path):
            for filename in files:
                if filename.endswith(".py") and not filename.startswith("_"):
                    file_path = os.path.join(root, filename)
                    relative_path = os.path.relpath(file_path, directory_path)
                    module_name = f"{base_module_name}.{os.path.splitext(relative_path)[0].replace(os.sep, '.')}"
                    process_file(file_path, module_name)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <directory_path> <base_module_name>")
        sys.exit(1)

    directory_path = sys.argv[1]
    base_module_name = sys.argv[2]
    process_directory(directory_path, base_module_name)
