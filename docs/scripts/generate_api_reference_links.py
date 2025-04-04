import argparse
import importlib
import inspect
import json
import logging
import os
import re
import warnings
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


# (alias/re-exported modules, source module, class, docs namespace)
MANUAL_API_REFERENCES_LANGGRAPH = [
    (
        ["langgraph.prebuilt"],
        "langgraph.prebuilt.chat_agent_executor",
        "create_react_agent",
        "prebuilt",
    ),
    (["langgraph.prebuilt"], "langgraph.prebuilt.tool_node", "ToolNode", "prebuilt"),
    (
        ["langgraph.prebuilt"],
        "langgraph.prebuilt.tool_node",
        "tools_condition",
        "prebuilt",
    ),
    (
        ["langgraph.prebuilt"],
        "langgraph.prebuilt.tool_node",
        "InjectedState",
        "prebuilt",
    ),
    # Graph
    (["langgraph.graph"], "langgraph.graph.message", "add_messages", "graphs"),
    (["langgraph.graph"], "langgraph.graph.state", "StateGraph", "graphs"),
    (["langgraph.graph"], "langgraph.graph.state", "CompiledStateGraph", "graphs"),
    ([], "langgraph.types", "StreamMode", "types"),
    (["langgraph.graph"], "langgraph.constants", "START", "constants"),
    (["langgraph.graph"], "langgraph.constants", "END", "constants"),
    (["langgraph.constants"], "langgraph.types", "Send", "types"),
    (["langgraph.constants"], "langgraph.types", "Interrupt", "types"),
    ([], "langgraph.types", "RetryPolicy", "types"),
    ([], "langgraph.checkpoint.base", "Checkpoint", "checkpoints"),
    ([], "langgraph.checkpoint.base", "CheckpointMetadata", "checkpoints"),
    ([], "langgraph.checkpoint.base", "BaseCheckpointSaver", "checkpoints"),
    ([], "langgraph.checkpoint.base", "SerializerProtocol", "checkpoints"),
    ([], "langgraph.checkpoint.serde.jsonplus", "JsonPlusSerializer", "checkpoints"),
    ([], "langgraph.checkpoint.memory", "MemorySaver", "checkpoints"),
    ([], "langgraph.checkpoint.sqlite.aio", "AsyncSqliteSaver", "checkpoints"),
    ([], "langgraph.checkpoint.sqlite", "SqliteSaver", "checkpoints"),
    ([], "langgraph.checkpoint.postgres.aio", "AsyncPostgresSaver", "checkpoints"),
    ([], "langgraph.checkpoint.postgres", "PostgresSaver", "checkpoints"),
]

WELL_KNOWN_LANGGRAPH_OBJECTS = {
    (module_, class_): (source_module, namespace)
    for (modules, source_module, class_, namespace) in MANUAL_API_REFERENCES_LANGGRAPH
    for module_ in modules + [source_module]
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
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            module = importlib.import_module(module_path)
        except ImportError:
            # check if it's a submodule
            try:
                module = importlib.import_module(module_path + "." + class_name)
            except ImportError:
                raise ValueError(f"Failed to import module {module_path}")
            return module.__name__

        try:
            class_ = getattr(module, class_name)
        except AttributeError:
            # check if it's a submodule
            try:
                module = importlib.import_module(module_path + "." + class_name)
            except ImportError:
                raise ValueError(
                    f"Failed to import class {class_name} from module {module_path}"
                )
            return module.__name__

        inspectmodule = inspect.getmodule(class_)
        if inspectmodule is None:
            # it wasn't a class, it's a primitive (e.g. END="__end__")
            # so no documentation link is necessary
            return None
        return inspectmodule.__name__


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
            try:
                module_path = get_full_module_name(module, class_name)
            except ValueError as e:
                logger.warning(e)
                continue
            except Exception as e:
                logger.error(
                    f"Failed to get full module name for {module}.{class_name}"
                )
                logger.error(e)
                continue
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
                if (module, class_name) not in WELL_KNOWN_LANGGRAPH_OBJECTS:
                    # Likely not documented yet
                    continue

                source_module, namespace = WELL_KNOWN_LANGGRAPH_OBJECTS[
                    (module, class_name)
                ]
                url = (
                    _LANGGRAPH_API_REFERENCE
                    + namespace
                    + "/#"
                    + source_module
                    + "."
                    + class_name
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
