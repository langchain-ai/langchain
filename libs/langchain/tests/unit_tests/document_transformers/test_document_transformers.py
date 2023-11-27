from typing import Sequence

import pytest

from langchain.document_transformers.document_transformers import (
    _LEGACY,
    DocumentTransformers,
)
from langchain.schema import Document
from langchain.text_splitter import CharacterTextSplitter

from .sample_transformer import LowerLazyTransformer, UpperLazyTransformer


def by_pg(doc: Document) -> str:
    return doc.page_content


@pytest.mark.skipif(not _LEGACY, reason="Test only runnable transformer")
@pytest.mark.parametrize(
    "transformers",
    [
        [
            CharacterTextSplitter(separator=" ", chunk_size=1, chunk_overlap=0),
            UpperLazyTransformer(),
        ]
    ],
)
def test_document_transformers_legacy(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")

    transformer = DocumentTransformers(transformers=transformers)
    r = transformer.transform_documents([doc1, doc2])
    assert len(r) == 6
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content="my"),
            Document(page_content="test"),
            Document(page_content=doc1.page_content.upper()),
            Document(page_content="other"),
            Document(page_content="test"),
            Document(page_content=doc2.page_content.upper()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
def test_transform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transformer = DocumentTransformers(transformers=transformers)
    r = transformer.transform_documents([doc1, doc2])
    assert len(r) == 4
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
@pytest.mark.asyncio
async def test_atransform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transformer = DocumentTransformers(transformers=transformers)
    r = await transformer.atransform_documents([doc1, doc2])
    assert len(r) == 4
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
def test_lazy_transform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transformer = DocumentTransformers(transformers=transformers)
    r = [doc for doc in transformer.lazy_transform_documents(iter([doc1, doc2]))]
    assert len(r) == 4

    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )


@pytest.mark.parametrize(
    "transformers", [[UpperLazyTransformer(), LowerLazyTransformer()]]
)
@pytest.mark.asyncio
async def test_alazy_transform_documents(transformers: Sequence) -> None:
    doc1 = Document(page_content="my test")
    doc2 = Document(page_content="other test")
    transformer = DocumentTransformers(transformers=transformers)
    r = [doc async for doc in transformer.alazy_transform_documents(iter([doc1, doc2]))]
    assert len(r) == 4
    assert sorted(r, key=by_pg) == sorted(
        [
            Document(page_content=doc1.page_content.upper()),
            Document(page_content=doc2.page_content.upper()),
            Document(page_content=doc1.page_content.lower()),
            Document(page_content=doc2.page_content.lower()),
        ],
        key=by_pg,
    )


def test_lcel_add_transform_documents() -> None:
    """Test create documents method."""
    x = UpperLazyTransformer()
    assert len(((x + x) + x).transformers) == 3
    assert len((x + (x + x)).transformers) == 3
    assert len(((x + x) + (x + x)).transformers) == 4
    assert len((x + x + x).transformers) == 3


@pytest.mark.skipif(not _LEGACY, reason="Test only runnable transformer")
def test_lcel_add_mixte_transform_documents() -> None:
    """Test create documents method."""
    x = UpperLazyTransformer()
    y = CharacterTextSplitter()
    assert len((y + x).transformers) == 2
    assert len((x + y).transformers) == 2
    assert len(((x + x) + y).transformers) == 3
    assert len((y + (x + x)).transformers) == 3
