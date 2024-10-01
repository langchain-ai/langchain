import argparse
import importlib
import inspect
import json
import logging
import os
import re
from pathlib import Path
from typing import List, Literal, Optional

from typing_extensions import TypedDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Base URL for all class documentation
_LANGCHAIN_API_REFERENCE = "https://python.langchain.com/api_reference/"
_LANGGRAPH_API_REFERENCE = "https://langchain-ai.github.io/langgraph/reference/"

# Regular expression to match Python code blocks
code_block_re = re.compile(r"^(```\s?python\n)(.*?)(```)", re.DOTALL | re.MULTILINE)


MANUAL_API_REFERENCES_LANGGRAPH = [
    ("langgraph.prebuilt", "create_react_agent"),
    (
        "langgraph.prebuilt",
        "ToolNode",
    ),
    (
        "langgraph.prebuilt",
        "ToolExecutor",
    ),
    (
        "langgraph.prebuilt",
        "ToolInvocation",
    ),
    ("langgraph.prebuilt", "tools_condition"),
    (
        "langgraph.prebuilt",
        "ValidationNode",
    ),
    (
        "langgraph.prebuilt",
        "InjectedState",
    ),
    # Graph
    (
        "langgraph.graph",
        "StateGraph",
    ),
    (
        "langgraph.graph.message",
        "MessageGraph",
    ),
    ("langgraph.graph.message", "add_messages"),
    (
        "langgraph.graph.graph",
        "CompiledGraph",
    ),
    (
        "langgraph.types",
        "StreamMode",
    ),
    (
        "langgraph.graph",
        "START",
    ),
    (
        "langgraph.graph",
        "END",
    ),
    (
        "langgraph.types",
        "Send",
    ),
    (
        "langgraph.types",
        "Interrupt",
    ),
    (
        "langgraph.types",
        "RetryPolicy",
    ),
    (
        "langgraph.checkpoint.base",
        "Checkpoint",
    ),
    (
        "langgraph.checkpoint.base",
        "CheckpointMetadata",
    ),
    (
        "langgraph.checkpoint.base",
        "BaseCheckpointSaver",
    ),
    (
        "langgraph.checkpoint.base",
        "SerializerProtocol",
    ),
    (
        "langgraph.checkpoint.serde.jsonplus",
        "JsonPlusSerializer",
    ),
    (
        "langgraph.checkpoint.memory",
        "MemorySaver",
    ),
    (
        "langgraph.checkpoint.sqlite.aio",
        "AsyncSqliteSaver",
    ),
    (
        "langgraph.checkpoint.sqlite",
        "SqliteSaver",
    ),
    (
        "langgraph.checkpoint.postgres.aio",
        "AsyncPostgresSaver",
    ),
    (
        "langgraph.checkpoint.postgres",
        "PostgresSaver",
    ),
]

WELL_KNOWN_LANGGRAPH_OBJECTS = {
    (module_, class_) for module_, class_ in MANUAL_API_REFERENCES_LANGGRAPH
}


def _make_regular_expression(pkg_prefix: str) -> re.Pattern:
    if not pkg_prefix.isidentifier():
        raise ValueError(f"Invalid package prefix: {pkg_prefix}")
    return re.compile(
        r"from\s+(" + pkg_prefix + "(?:_\w+)?(?:\.\w+)*?)\s+import\s+"
        r"((?:\w+(?:,\s*)?)*"  # Match zero or more words separated by a comma+optional ws
        r"(?:\s*\(.*?\))?)",  # Match optional parentheses block
        re.DOTALL,  # Match newlines as well
    )


# Regular expression to match langchain import lines
_IMPORT_LANGCHAIN_RE = _make_regular_expression("langchain")
_IMPORT_LANGGRAPH_RE = _make_regular_expression("langgraph")


_CURRENT_PATH = Path(__file__).parent.absolute()
# Directory where generated markdown files are stored
_DOCS_DIR = _CURRENT_PATH.parent.parent / "docs"


def find_files(path):
    """Find all MDX files in the given path"""
    # Check if is file first
    if ".ipynb_checkpoints" in str(path):
        return
    if os.path.isfile(path):
        yield path
        return
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".mdx") or file.endswith(".md"):
                full = os.path.join(root, file)
                if ".ipynb_checkpoints" in str(full):
                    continue
                yield full


def get_full_module_name(module_path, class_name) -> Optional[str]:
    """Get full module name using inspect"""
    try:
        module = importlib.import_module(module_path)
        class_ = getattr(module, class_name)
        return inspect.getmodule(class_).__name__
    except AttributeError as e:
        logger.warning(f"Could not find module for {class_name}, {e}")
        return None
    except ImportError as e:
        logger.warning(f"Failed to load for class {class_name}, {e}")
        return None


def get_args() -> argparse.Namespace:
    """Get command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--docs_dir",
        type=str,
        default=_DOCS_DIR,
        help="Directory where generated markdown files are stored",
    )
    parser.add_argument(
        "--json_path",
        type=str,
        default=None,
        help="Path to store the generated JSON file",
    )
    return parser.parse_args()


def main() -> None:
    """Main function"""
    args = get_args()
    global_imports = {}

    for file in find_files(args.docs_dir):
        file_imports = replace_imports(file)

        if file_imports:
            # Use relative file path as key
            relative_path = (
                os.path.relpath(file, args.docs_dir)
                .replace(".mdx", "/")
                .replace(".md", "/")
            )

            doc_url = f"https://python.langchain.com/docs/{relative_path}"
            for import_info in file_imports:
                doc_title = import_info["title"]
                class_name = import_info["imported"]
                if class_name not in global_imports:
                    global_imports[class_name] = {}
                global_imports[class_name][doc_title] = doc_url

    # Write the global imports information to a JSON file
    if args.json_path:
        json_path = Path(args.json_path)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        with json_path.open("w") as f:
            json.dump(global_imports, f)


def _get_doc_title(data: str, file_name: str) -> str:
    try:
        return re.findall(r"^#\s*(.*)", data, re.MULTILINE)[0]
    except IndexError:
        pass
    # Parse the rst-style titles
    try:
        return re.findall(r"^(.*)\n=+\n", data, re.MULTILINE)[0]
    except IndexError:
        return file_name


class ImportInformation(TypedDict):
    imported: str  # imported class name
    source: str  # module path
    docs: str  # URL to the documentation
    title: str  # Title of the document


def _get_imports(
    code: str, doc_title: str, package_ecosystem: Literal["langchain", "langgraph"]
) -> List[ImportInformation]:
    """Get imports from the given code block.

    Args:
        code: Python code block from which to extract imports
        doc_title: Title of the document
        package_ecosystem: "langchain" or "langgraph". The two live in different
            repositories and have separate documentation sites.

    Returns:
        List of import information for the given code block
    """
    imports = []

    if package_ecosystem == "langchain":
        pattern = _IMPORT_LANGCHAIN_RE
    elif package_ecosystem == "langgraph":
        pattern = _IMPORT_LANGGRAPH_RE
    else:
        raise ValueError(f"Invalid package ecosystem: {package_ecosystem}")

    for import_match in pattern.finditer(code):
        module = import_match.group(1)
        if "pydantic_v1" in module:
            continue
        imports_str = (
            import_match.group(2).replace("(\n", "").replace("\n)", "")
        )  # Handle newlines within parentheses
        # remove any newline and spaces, then split by comma
        imported_classes = [
            imp.strip()
            for imp in re.split(r",\s*", imports_str.replace("\n", ""))
            if imp.strip()
        ]
        for class_name in imported_classes:
            module_path = get_full_module_name(module, class_name)
            if not module_path:
                continue
            if len(module_path.split(".")) < 2:
                continue

            if package_ecosystem == "langchain":
                pkg = module_path.split(".")[0].replace("langchain_", "")
                top_level_mod = module_path.split(".")[1]

                url = (
                    _LANGCHAIN_API_REFERENCE
                    + pkg
                    + "/"
                    + top_level_mod
                    + "/"
                    + module_path
                    + "."
                    + class_name
                    + ".html"
                )
            elif package_ecosystem == "langgraph":
                if module.startswith("langgraph.checkpoint"):
                    namespace = "checkpoints"
                elif module.startswith("langgraph.graph"):
                    namespace = "graphs"
                elif module.startswith("langgraph.prebuilt"):
                    namespace = "prebuilt"
                elif module.startswith("langgraph.errors"):
                    namespace = "errors"
                else:
                    # Likely not documented yet
                    # Unable to determine the namespace
                    continue

                if module.startswith("langgraph.errors"):
                    # Has different URL structure than other modules
                    url = (
                        _LANGGRAPH_API_REFERENCE
                        + namespace
                        + "/#langgraph.errors."
                        + class_name  # Uses the actual class name here.
                    )
                else:
                    if (module, class_name) not in WELL_KNOWN_LANGGRAPH_OBJECTS:
                        # Likely not documented yet
                        continue
                    url = (
                        _LANGGRAPH_API_REFERENCE + namespace + "/#" + class_name.lower()
                    )
            else:
                raise ValueError(f"Invalid package ecosystem: {package_ecosystem}")

            # Add the import information to our list
            imports.append(
                {
                    "imported": class_name,
                    "source": module,
                    "docs": url,
                    "title": doc_title,
                }
            )

    return imports


def replace_imports(file) -> List[ImportInformation]:
    """Replace imports in each Python code block with links to their
    documentation and append the import info in a comment

    Returns:
        list of import information for the given file
    """
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

        imports.extend(_get_imports(code, _DOC_TITLE, "langchain"))
        imports.extend(_get_imports(code, _DOC_TITLE, "langgraph"))

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
