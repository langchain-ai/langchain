import os
from pathlib import Path
from typing import Callable

import pytest

from langchain_unstructured import UnstructuredLoader

EXAMPLE_DOCS_DIRECTORY = str(
    Path(__file__).parent.parent.parent.parent.parent
    / "community/tests/integration_tests/examples/"
)
UNSTRUCTURED_API_KEY = os.getenv("UNSTRUCTURED_API_KEY")


# -- Local partition --


@pytest.mark.local
def test_loader_partitions_locally() -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    docs = UnstructuredLoader(
        file_path=file_path,
        # Unstructured kwargs
        strategy="fast",
        include_page_breaks=True,
    ).load()

    assert all(
        doc.metadata.get("filename") == "layout-parser-paper.pdf" for doc in docs
    )
    assert any(doc.metadata.get("category") == "PageBreak" for doc in docs)


@pytest.mark.local
def test_loader_partition_ignores_invalid_arg() -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")

    docs = UnstructuredLoader(
        file_path=file_path,
        # Unstructured kwargs
        strategy="fast",
        # mode is no longer a valid argument and is ignored when partitioning locally
        mode="single",
    ).load()

    assert len(docs) > 1
    assert all(
        doc.metadata.get("filename") == "layout-parser-paper.pdf" for doc in docs
    )


@pytest.mark.local
def test_loader_partitions_locally_and_applies_post_processors(
    get_post_processor: Callable[[str], str],
) -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredLoader(
        file_path=file_path,
        post_processors=[get_post_processor],
        strategy="fast",
    )

    docs = loader.load()

    assert len(docs) > 1
    assert docs[0].page_content.endswith("THE END!")


# -- API partition --


def test_loader_partitions_via_api() -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredLoader(
        file_path=file_path,
        partition_via_api=True,
        # Unstructured kwargs
        strategy="fast",
        include_page_breaks=True,
    )

    docs = loader.load()

    assert len(docs) > 1
    assert any(doc.metadata.get("category") == "PageBreak" for doc in docs)
    assert all(
        doc.metadata.get("filename") == "layout-parser-paper.pdf" for doc in docs
    )
    assert docs[0].metadata.get("element_id") is not None


def test_loader_partitions_multiple_via_api() -> None:
    file_paths = [
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf"),
        os.path.join(EXAMPLE_DOCS_DIRECTORY, "fake-email-attachment.eml"),
    ]
    loader = UnstructuredLoader(
        file_path=file_paths,
        api_key=UNSTRUCTURED_API_KEY,
        partition_via_api=True,
        # Unstructured kwargs
        strategy="fast",
    )

    docs = loader.load()

    assert len(docs) > 1
    assert docs[0].metadata.get("filename") == "layout-parser-paper.pdf"
    assert docs[-1].metadata.get("filename") == "fake-email-attachment.eml"


def test_loader_partition_via_api_raises_TypeError_with_invalid_arg() -> None:
    file_path = os.path.join(EXAMPLE_DOCS_DIRECTORY, "layout-parser-paper.pdf")
    loader = UnstructuredLoader(
        file_path=file_path,
        api_key=UNSTRUCTURED_API_KEY,
        partition_via_api=True,
        mode="elements",
    )

    with pytest.raises(TypeError, match="unexpected keyword argument 'mode'"):
        loader.load()


# -- fixtures ---


@pytest.fixture()
def get_post_processor() -> Callable[[str], str]:
    def append_the_end(text: str) -> str:
        return text + "THE END!"

    return append_the_end
