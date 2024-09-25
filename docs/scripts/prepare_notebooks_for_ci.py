"""Preprocess notebooks for CI. Currently adds VCR cassettes and optionally removes pip install cells."""

import logging
import os
import json
import click
import nbformat

logger = logging.getLogger(__name__)
NOTEBOOK_DIRS = ("docs/docs/how-tos","docs/docs/tutorials")
DOCS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASSETTES_PATH = os.path.join(DOCS_PATH, "cassettes")

# TODO: update these
NOTEBOOKS_NO_CASSETTES = (
    "docs/docs/how-tos/visualization.ipynb",
    "docs/docs/how-tos/many-tools.ipynb"
)

NOTEBOOKS_NO_EXECUTION = [
    "docs/docs/tutorials/local_rag.ipynb",
]


def comment_install_cells(notebook: nbformat.NotebookNode) -> nbformat.NotebookNode:
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue

        if "pip install" in cell.source:
            # Comment out the lines in cells containing "pip install"
            cell.source = "\n".join(
                f"# {line}" if line.strip() else line
                for line in cell.source.splitlines()
            )

    return notebook


def is_magic_command(code: str) -> bool:
    return code.strip().startswith("%") or code.strip().startswith("!")


def is_comment(code: str) -> bool:
    return code.strip().startswith("#")


def add_vcr_to_notebook(
    notebook: nbformat.NotebookNode, cassette_prefix: str
) -> nbformat.NotebookNode:
    """Inject `with vcr.cassette` into each code cell of the notebook."""

    # Inject VCR context manager into each code cell
    for idx, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue

        lines = cell.source.splitlines()
        # skip if empty cell
        if not lines:
            continue

        are_magic_lines = [is_magic_command(line) for line in lines]

        # skip if all magic
        if all(are_magic_lines):
            continue

        if any(are_magic_lines):
            raise ValueError(
                "Cannot process code cells with mixed magic and non-magic code."
            )

        # skip if just comments
        if all(is_comment(line) or not line.strip() for line in lines):
            continue

        cell_id = cell.get("id", idx)
        cassette_name = f"{cassette_prefix}_{cell_id}.msgpack.zlib"
        cell.source = f"with custom_vcr.use_cassette('{cassette_name}', filter_headers=['x-api-key', 'authorization'], record_mode='once', serializer='advanced_compressed'):\n" + "\n".join(
            f"    {line}" for line in lines
        )

    # Add import statement
    vcr_import_lines = [
        "import vcr",
        "import msgpack",
        "import nest_asyncio",
        "import base64",
        "import zlib",
        "import re",
        "",
        "custom_vcr = vcr.VCR()",
        "",
        "def compress_data(data, compression_level=9):",
        "    packed = msgpack.packb(data, use_bin_type=True)",
        "    compressed = zlib.compress(packed, level=compression_level)",
        "    return base64.b64encode(compressed).decode('utf-8')",
        "",
        "def decompress_data(compressed_string):",
        "    try:",
        "        decoded = base64.b64decode(compressed_string)",
        "        decompressed = zlib.decompress(decoded)",
        "        return msgpack.unpackb(decompressed, raw=False)",
        "    except (ValueError, zlib.error, msgpack.exceptions.ExtraData, msgpack.exceptions.UnpackValueError):",
        "        return {\"requests\": [], \"responses\": []}",
        "",
        "def filter_cassette_data(cassette_dict):",
        "    for interaction in cassette_dict['interactions']:",
        "        if len(interaction['response']['body']['string']) > 1000:",
        "            interaction['response']['body']['string'] = interaction['response']['body']['string'][:1000] + '... (truncated)'",
        "        for req_or_res in [interaction['request'], interaction['response']]:",
        "            headers_to_remove = ['date', 'server', 'content-length']",
        "            for header in headers_to_remove:",
        "                req_or_res['headers'].pop(header, None)",
        "    return cassette_dict",
        "",
        "class AdvancedCompressedSerializer:",
        "    def serialize(self, cassette_dict):",
        "        filtered_dict = filter_cassette_data(cassette_dict)",
        "        return compress_data(filtered_dict)",
        "",
        "    def deserialize(self, cassette_string):",
        "        return decompress_data(cassette_string)",
        "",
        "custom_vcr.register_serializer('advanced_compressed', AdvancedCompressedSerializer())",
        "",
        "def custom_matcher(r1, r2):",
        "    return (r1.method == r2.method and",
        "            r1.url == r2.url and",
        "            normalize_body(r1.body) == normalize_body(r2.body))",
        "",
        "def normalize_body(body):",
        "    return re.sub(r'\\s+', '', body.lower()) if body else ''",
        "",
        "nest_asyncio.apply()",
        "",
        "custom_vcr.serializer = 'advanced_compressed'",
        "custom_vcr.record_mode = 'new_episodes'",
        "custom_vcr.match_on = ['custom']",
        "custom_vcr.register_matcher('custom', custom_matcher)",
        "custom_vcr.filter_headers = ['authorization', 'user-agent', 'date', 'server']",
        "custom_vcr.filter_post_data_parameters = ['password', 'token']",
    ]
    import_cell = nbformat.v4.new_code_cell(source="\n".join(vcr_import_lines))
    import_cell.pop("id", None)
    notebook.cells.insert(0, import_cell)
    return notebook


def process_notebooks(should_comment_install_cells: bool) -> None:
    for directory in NOTEBOOK_DIRS:
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith(".ipynb") or "ipynb_checkpoints" in root:
                    continue

                notebook_path = os.path.join(root, file)
                try:
                    notebook = nbformat.read(notebook_path, as_version=4)

                    if should_comment_install_cells:
                        notebook = comment_install_cells(notebook)

                    base_filename = os.path.splitext(os.path.basename(file))[0]
                    cassette_prefix = os.path.join(CASSETTES_PATH, base_filename)
                    if notebook_path not in NOTEBOOKS_NO_CASSETTES:
                        notebook = add_vcr_to_notebook(
                            notebook, cassette_prefix=cassette_prefix
                        )

                    if notebook_path in NOTEBOOKS_NO_EXECUTION:
                        # Add a cell at the beginning to indicate that this notebook should not be executed
                        warning_cell = nbformat.v4.new_markdown_cell(
                            source="**Warning:** This notebook is not meant to be executed automatically."
                        )
                        notebook.cells.insert(0, warning_cell)

                        # Add a special tag to the first code cell
                        if notebook.cells and notebook.cells[1].cell_type == "code":
                            notebook.cells[1].metadata["tags"] = notebook.cells[1].metadata.get("tags", []) + ["no_execution"]

                    nbformat.write(notebook, notebook_path)
                    logger.info(f"Processed: {notebook_path}")
                except Exception as e:
                    logger.error(f"Error processing {notebook_path}: {e}")
    
    with open(os.path.join(DOCS_PATH, "notebooks_no_execution.json"), "w") as f:
        json.dump(NOTEBOOKS_NO_EXECUTION, f)


@click.command()
@click.option(
    "--comment-install-cells",
    is_flag=True,
    default=False,
    help="Whether to comment out install cells",
)
def main(comment_install_cells):
    process_notebooks(should_comment_install_cells=comment_install_cells)
    logger.info("All notebooks processed successfully.")


if __name__ == "__main__":
    main()
