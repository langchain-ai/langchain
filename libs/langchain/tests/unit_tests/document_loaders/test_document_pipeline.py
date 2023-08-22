"""Test simple document pipeline."""
from typing import Any, Iterator, List, Sequence

import pytest

from langchain.document_loaders.base import BaseLoader
from langchain.document_loaders.pipeline import DocumentPipeline
from langchain.schema import BaseDocumentTransformer, Document


class ToyLoader(BaseLoader):
    """Toy loader that always returns the same documents."""

    def __init__(self, documents: Sequence[Document]) -> None:
        """Initialize with the documents to return."""
        self.documents = documents

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        yield from self.documents

    def load(self) -> List[Document]:
        """Load the documents from the source."""
        return list(self.lazy_load())


class SimpleSplitter(BaseDocumentTransformer):
    def __init__(self, sentinel: int) -> None:
        """Initialize with the sentinel value."""
        self.sentinel = sentinel

    def transform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        """Split the document into two documents."""
        docs = []
        for document in documents:
            doc1 = document.copy()
            doc1.page_content = doc1.page_content + f"({self.sentinel}|1)"
            docs.append(doc1)

            doc2 = document.copy()
            doc2.page_content = doc2.page_content + f"({self.sentinel}|2)"
            docs.append(doc2)
        return docs

    async def atransform_documents(
        self, documents: Sequence[Document], **kwargs: Any
    ) -> Sequence[Document]:
        raise NotImplementedError()


@pytest.fixture
def loader() -> ToyLoader:
    """Get a toy loader."""
    return ToyLoader(
        documents=[
            Document(
                page_content="A",
            ),
            Document(
                page_content="B",
            ),
        ]
    )


def test_methods_should_remain_unimplemented(loader: ToyLoader) -> None:
    """Test the document pipeline."""
    pipeline = DocumentPipeline(loader)
    with pytest.raises(NotImplementedError):
        pipeline.load()
    with pytest.raises(NotImplementedError):
        pipeline.load_and_split()


def test_simple_pipeline(loader: ToyLoader) -> None:
    """Test simple document pipeline."""
    pipeline = DocumentPipeline(loader)
    assert list(pipeline.lazy_load()) == loader.documents


def test_pipeline_with_transformations(loader: ToyLoader) -> None:
    """Test pipeline with transformations."""
    pipeline = DocumentPipeline(
        loader, transformers=[SimpleSplitter(1), SimpleSplitter(2)]
    )

    docs = list(pipeline.lazy_load())

    assert sorted(doc.page_content for doc in docs) == [
        "A(1|1)(2|1)",
        "A(1|1)(2|2)",
        "A(1|2)(2|1)",
        "A(1|2)(2|2)",
        "B(1|1)(2|1)",
        "B(1|1)(2|2)",
        "B(1|2)(2|1)",
        "B(1|2)(2|2)",
    ]
