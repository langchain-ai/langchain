"""Preprocess notebooks for CI. Currently adds VCR cassettes and optionally removes pip install cells."""

import json
import logging
import os

import click
import nbformat

logger = logging.getLogger(__name__)
NOTEBOOK_DIRS = ("docs/docs/how_to", "docs/docs/tutorials")
DOCS_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CASSETTES_PATH = os.path.join(DOCS_PATH, "cassettes")

NOTEBOOKS_NO_CASSETTES = [
    "docs/docs/tutorials/rag.ipynb",  # TODO: accommodate langsmith changes in cassettes
    "docs/docs/tutorials/retrievers.ipynb",  # TODO: fix non-determinism
    "docs/docs/how_to/multi_vector.ipynb",  # Non-determinism due to batch
    "docs/docs/how_to/qa_sources.ipynb",  # TODO: accommodate langsmith changes in cassettes
    "docs/docs/how_to/qa_streaming.ipynb",  # TODO: accommodate langsmith changes in cassettes
]

NOTEBOOKS_NO_EXECUTION = [
    "docs/docs/how_to/add_scores_retriever.ipynb",  # Requires Pinecone instance
    "docs/docs/how_to/chat_model_rate_limiting.ipynb",  # Slow (demonstrates rate limiting)
    "docs/docs/how_to/document_loader_directory.ipynb",  # Deliberately raises error
    "docs/docs/how_to/document_loader_pdf.ipynb",  # Local parsing section is slow
    "docs/docs/how_to/example_selectors_langsmith.ipynb",  # TODO: add langchain-benchmarks; fix cassette issue
    "docs/docs/how_to/extraction_long_text.ipynb",  # Non-determinism due to batch
    "docs/docs/how_to/graph_constructing.ipynb",  # Requires local neo4j
    "docs/docs/how_to/graph_semantic.ipynb",  # Requires local neo4j
    "docs/docs/how_to/hybrid.ipynb",  # Requires AstraDB instance
    "docs/docs/how_to/indexing.ipynb",  # Requires local Elasticsearch
    "docs/docs/how_to/local_llms.ipynb",  # Local LLMs
    "docs/docs/how_to/migrate_agent.ipynb",  # TODO: resolve issue with asyncio / exception handling
    "docs/docs/how_to/qa_per_user.ipynb",  # Requires Pinecone instance
    "docs/docs/how_to/query_high_cardinality.ipynb",  # Heavy
    "docs/docs/how_to/response_metadata.ipynb",  # Auth is annoying
    "docs/docs/how_to/split_by_token.ipynb",  # TODO: requires Korean document, also heavy deps
    "docs/docs/how_to/tools_error.ipynb",  # Deliberately raises error
    "docs/docs/how_to/tools_human.ipynb",  # Requires human input()
    "docs/docs/how_to/tools_prompting.ipynb",  # Local LLMs
    "docs/docs/tutorials/graph.ipynb",  # Requires local graph db running
    "docs/docs/tutorials/summarization.ipynb",  # TODO: source of non-determinism somewhere, fix or add to no cassettes
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
        cell.source = (
            f"with custom_vcr.use_cassette('{cassette_name}', filter_headers=['x-api-key', 'authorization'], record_mode='once', serializer='advanced_compressed'):\n"
            + "\n".join(f"    {line}" for line in lines)
        )

    # Add import statement
    vcr_import_lines = [
        "import nest_asyncio",
        "nest_asyncio.apply()",
        "import vcr",
        "import msgpack",
        "import base64",
        "import zlib",
        "custom_vcr = vcr.VCR()",
        "",
        "def compress_data(data, compression_level=9):",
        "    packed = msgpack.packb(data, use_bin_type=True)",
        "    compressed = zlib.compress(packed, level=compression_level)",
        "    return base64.b64encode(compressed).decode('utf-8')",
        "",
        "def decompress_data(compressed_string):",
        "    decoded = base64.b64decode(compressed_string)",
        "    decompressed = zlib.decompress(decoded)",
        "    return msgpack.unpackb(decompressed, raw=False)",
        "",
        "class AdvancedCompressedSerializer:",
        "    def serialize(self, cassette_dict):",
        "        return compress_data(cassette_dict)",
        "",
        "    def deserialize(self, cassette_string):",
        "        return decompress_data(cassette_string)",
        "",
        "custom_vcr.register_serializer('advanced_compressed', AdvancedCompressedSerializer())",
        "custom_vcr.serializer = 'advanced_compressed'",
    ]
    import_cell = nbformat.v4.new_code_cell(source="\n".join(vcr_import_lines))
    import_cell.pop("id", None)
    notebook.cells.insert(0, import_cell)
    return notebook


def process_notebooks(
    should_comment_install_cells: bool,
    working_directory: str,
) -> None:
    for directory in NOTEBOOK_DIRS:
        for root, _, files in os.walk(directory):
            for file in files:
                if not file.endswith(".ipynb") or "ipynb_checkpoints" in root:
                    continue

                notebook_path = os.path.join(root, file)
                # Filter notebooks based on the working_directory input
                if working_directory != "all" and not notebook_path.startswith(
                    working_directory
                ):
                    continue

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
                            notebook.cells[1].metadata["tags"] = notebook.cells[
                                1
                            ].metadata.get("tags", []) + ["no_execution"]

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
@click.option(
    "--working-directory",
    default="all",
    help="Working directory or specific notebook to process",
)
def main(comment_install_cells, working_directory):
    process_notebooks(
        should_comment_install_cells=comment_install_cells,
        working_directory=working_directory,
    )
    logger.info("All notebooks processed successfully.")


if __name__ == "__main__":
    main()
