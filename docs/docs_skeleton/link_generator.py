import importlib
import inspect
import logging
import os
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Base URL for all class documentation
base_url = "https://api.python.langchain.com/en/latest/"

# Directory where generated markdown files are stored
docs_dir = "docs_skeleton/docs"

# Regular expression to match Python import lines
import_re = re.compile(r"(from\s+(langchain\.\w+(\.\w+)*?)\s+import\s+)(\w+)")


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


def replace_imports(file):
    """Replace imports in a markdown file with links to their documentation"""
    with open(file, "r") as f:
        data = f.read()

    for match in import_re.finditer(data):
        # Create the URL for this class's documentation
        class_name = match.group(4)
        try:
            module_path = get_full_module_name(match.group(2), class_name)
        except AttributeError as e:
            logger.error(f"Could not find module for {class_name}", e)
            continue
        module_path_parts = module_path.replace(".", "/")

        url = (
            base_url
            + module_path_parts
            + "/"
            + module_path_parts
            + "."
            + class_name
            + ".html"
        )

        # Replace the class name with a markdown link
        link = f"[{match.group(1)}{class_name}]({url})"
        data = data.replace(match.group(0), f"{match.group(0)}\n{link}")

    with open(file, "w") as f:
        f.write(data)


def main():
    """Main function"""
    for file in find_files(docs_dir):
        print(f"Adding links for imports in {file}")
        replace_imports(file)


if __name__ == "__main__":
    main()
