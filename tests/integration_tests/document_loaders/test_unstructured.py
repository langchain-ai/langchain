import os
from contextlib import ExitStack
from pathlib import Path

from langchain.document_loaders import (
    UnstructuredAPIFileIOLoader,
    UnstructuredAPIFileLoader,
)

EXAMPLE_DOCS_DIRECTORY = str(Path(__file__).parent.parent / "examples/")


def test_unstructured_api_file_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredAPIFileLoader(
        file_path=file_path,
        api_key="FAKE_API_KEY",
        strategy="fast",
        mode="elements",
    )
    docs = loader.load()

    assert len(docs) > 1


def test_unstructured_api_file_loader_multiple_files() -> None:
    """Test unstructured loader."""
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
    ]

    loader = UnstructuredAPIFileLoader(
        file_path=file_paths,
        api_key="FAKE_API_KEY",
        strategy="fast",
        mode="elements",
    )
    docs = loader.load()

    assert len(docs) > 1


def test_unstructured_api_file_io_loader() -> None:
    """Test unstructured loader."""
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    with open(file_path, "rb") as f:
        loader = UnstructuredAPIFileIOLoader(
            file=f,
            api_key="FAKE_API_KEY",
            strategy="fast",
            mode="elements",
            file_filename=file_path,
        )
        docs = loader.load()

    assert len(docs) > 1


def test_unstructured_api_file_loader_io_multiple_files() -> None:
    """Test unstructured loader."""
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "whatsapp_chat.txt"),
    ]

    with ExitStack() as stack:
        files = [stack.enter_context(open(file_path, "rb")) for file_path in file_paths]

        loader = UnstructuredAPIFileIOLoader(
            file=files,  # type: ignore
            api_key="FAKE_API_KEY",
            strategy="fast",
            mode="elements",
            file_filenames=file_paths,
        )

        docs = loader.load()

    assert len(docs) > 1
