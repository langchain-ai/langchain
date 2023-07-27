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

# Regular expression to match Python code blocks
code_block_re = re.compile(r"^(```python\n)(.*?)(```\n)", re.DOTALL | re.MULTILINE)
# Regular expression to match langchain import lines
_IMPORT_RE = re.compile(r"(from\s+(langchain\.\w+(\.\w+)*?)\s+import\s+)(\w+)")

_CURRENT_PATH = Path(__file__).parent.absolute()
# Directory where generated markdown files are stored
_DOCS_DIR = _CURRENT_PATH / "docs"
_JSON_PATH = _CURRENT_PATH.parent / "api_reference" / "guide_imports.json"


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
            doc_url = f"https://python.langchain.com/docs/{relative_path.replace('.mdx', '').replace('.md', '')}"
            for import_info in file_imports:
                doc_title = import_info["title"]
                class_name = import_info["imported"]
                if class_name not in global_imports:
                    global_imports[class_name] = {}
                global_imports[class_name][doc_title] = doc_url

    # Write the global imports information to a JSON file
    _JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _JSON_PATH.open("w") as f:
        json.dump(global_imports, f)


def _get_doc_title(data: str, file_name: str) -> str:
    try:
        return re.findall(r"^#\s+(.*)", data, re.MULTILINE)[0]
    except IndexError:
        pass
    # Parse the rst-style titles
    try:
        return re.findall(r"^(.*)\n=+\n", data, re.MULTILINE)[0]
    except IndexError:
        return file_name


def replace_imports(file):
    """Replace imports in each Python code block with links to their documentation and append the import info in a comment"""
    all_imports = []
    with open(file, "r") as f:
        data = f.read()

    file_name = os.path.basename(file)
    _DOC_TITLE = _get_doc_title(data, file_name)

    def replacer(match):
        # Extract the code block content
        code = match.group(2)
        # Replace if any import comment exists
        # TODO: Use our own custom <code> component rather than this
        # injection method
        existing_comment_re = re.compile(r"^<!--IMPORTS:.*?-->\n", re.MULTILINE)
        code = existing_comment_re.sub("", code)

        # Process imports in the code block
        imports = []
        for import_match in _IMPORT_RE.finditer(code):
            class_name = import_match.group(4)
            try:
                module_path = get_full_module_name(import_match.group(2), class_name)
            except AttributeError as e:
                logger.warning(f"Could not find module for {class_name}, {e}")
                continue
            except ImportError as e:
                # Some CentOS OpenSSL issues can cause this to fail
                logger.warning(f"Failed to load for class {class_name}, {e}")
                continue

            url = (
                _BASE_URL
                + "/"
                + module_path.split(".")[1]
                + "/"
                + module_path
                + "."
                + class_name
                + ".html"
            )

            # Add the import information to our list
            imports.append(
                {
                    "imported": class_name,
                    "source": import_match.group(2),
                    "docs": url,
                    "title": _DOC_TITLE,
                }
            )

        if imports:
            all_imports.extend(imports)
            # Create a unique comment containing the import information
            import_comment = f"<!--IMPORTS:{json.dumps(imports)}-->"
            # Inject the import comment at the start of the code block
            return match.group(1) + import_comment + "\n" + code + match.group(3)
        else:
            # If there are no imports, return the original match
            return match.group(0)

    # Use re.sub to replace each Python code block
    data = code_block_re.sub(replacer, data)

    with open(file, "w") as f:
        f.write(data)
    return all_imports


if __name__ == "__main__":
    main()
