import importlib
import inspect
import json
import logging
import os
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Base URL for all class documentation
_BASE_URL = "https://api.python.langchain.com/en/latest/"

# Regular expression to match Python import lines
_IMPORT_RE = re.compile(r"(from\s+(langchain\.\w+(\.\w+)*?)\s+import\s+)(\w+)")

_CODEBLOCK_PATH = "src/theme/CodeBlock/"
_CURRENT_PATH = Path(__file__).parent.absolute()
# Directory where generated markdown files are stored
_DOCS_DIR = _CURRENT_PATH / "docs"
# Will dump to the codeblock directory / imports.json
_JSON_PATH = _CURRENT_PATH / _CODEBLOCK_PATH / "imports.json"


def find_files(path):
    """Find all MDX files in the given path"""
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".mdx") or file.endswith(".md"):
                yield os.path.join(root, file)


def get_full_module_name(module_path, class_name):
    """Get full module name using inspect"""
    module = importlib.import_module(module_path)
    class_ = getattr(module, class_name)
    return inspect.getmodule(class_).__name__


def main():
    """Main function"""
    global_imports = {}

    for file in find_files(_DOCS_DIR):
        print(f"Adding links for imports in {file}")

        # replace_imports now returns the import information rather than writing it to a file
        file_imports = replace_imports(file)

        if file_imports:
            # Use relative file path as key
            relative_path = os.path.relpath(file, _DOCS_DIR)
            global_imports[relative_path] = file_imports

    # Write the global imports information to a JSON file
    with _JSON_PATH.open("w") as f:
        json.dump(global_imports, f)


def replace_imports(file):
    """Replace imports in a markdown file with links to their documentation and return the import info"""
    imports = []

    with open(file, "r") as f:
        data = f.read()

    for match in _IMPORT_RE.finditer(data):
        class_name = match.group(4)
        try:
            module_path = get_full_module_name(match.group(2), class_name)
        except AttributeError as e:
            logger.error(f"Could not find module for {class_name}", e)
            continue
        module_path_parts = module_path.replace(".", "/")

        url = (
            _BASE_URL
            + module_path_parts
            + "/"
            + module_path_parts
            + "."
            + class_name
            + ".html"
        )

        # Add the import information to our list
        imports.append({"imported": class_name, "source": match.group(2), "docs": url})

    return imports


if __name__ == "__main__":
    main()
