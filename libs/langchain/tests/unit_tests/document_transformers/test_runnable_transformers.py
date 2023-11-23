from typing import Tuple, Type

import pytest

import langchain
from langchain.document_transformers.runnable_document_transformer import (
    to_async_iterator,
)
from tests.unit_tests.document_transformers.sample_transformer import (
    LowerLazyTransformer,
    UpperLazyTransformer,
)


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
def test_transform(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = cls().transform_documents(documents=[doc1, doc2])
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_atransform(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = await cls().atransform_documents(documents=[doc1, doc2])
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
def test_lazy_transform(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = list(cls().lazy_transform_documents(documents=iter([doc1, doc2])))
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_alazy_transform_sync_iterator(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = [
        doc
        async for doc in cls().alazy_transform_documents(documents=iter([doc1, doc2]))
    ]
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_alazy_transform_async_iterator(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = [
        doc
        async for doc in cls().alazy_transform_documents(
            documents=to_async_iterator(iter([doc1, doc2]))
        )
    ]
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
def test_invoke_transformer(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = list(cls().invoke(iter([doc1, doc2])))
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_ainvoke_transformer_sync_iterator(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = [doc async for doc in await cls().ainvoke(iter([doc1, doc2]))]
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


@pytest.mark.parametrize("cls", [UpperLazyTransformer])
@pytest.mark.asyncio
async def test_ainvoke_transformer_async_iterator(cls: Type) -> None:
    doc1 = langchain.schema.Document(page_content="my test")
    doc2 = langchain.schema.Document(page_content="other test")
    r = [
        doc async for doc in await cls().ainvoke(to_async_iterator(iter([doc1, doc2])))
    ]
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.upper()),
        langchain.schema.Document(page_content=doc2.page_content.upper()),
    ]


# %%
@pytest.mark.parametrize(
    "cls",
    [
        (UpperLazyTransformer, LowerLazyTransformer),
    ],
)
def test_invoke_pipeline(cls: Tuple[Type, Type]) -> None:
    doc1 = langchain.schema.Document(page_content="My test")
    doc2 = langchain.schema.Document(page_content="Other test")
    runnable = cls[0]() | cls[1]()  # LCEL syntax
    r = list(runnable.invoke([doc1, doc2]))
    assert r == [
        langchain.schema.Document(page_content=doc1.page_content.lower()),
        langchain.schema.Document(page_content=doc2.page_content.lower()),
    ]


@pytest.mark.parametrize("cls", [(UpperLazyTransformer, LowerLazyTransformer)])
@pytest.mark.asyncio
async def test_ainvoke_pipeline_sync_iterator(cls: Tuple[Type, Type]) -> None:
    doc1 = langchain.schema.Document(page_content="My test")
    doc2 = langchain.schema.Document(page_content="Other test")
    runnable = cls[0]() | cls[1]()
    r = await runnable.ainvoke(to_async_iterator(iter([doc1, doc2])))
    rr = [doc async for doc in r]
    assert rr == [
        langchain.schema.Document(page_content=doc1.page_content.lower()),
        langchain.schema.Document(page_content=doc2.page_content.lower()),
    ]


@pytest.mark.parametrize("cls", [(UpperLazyTransformer, LowerLazyTransformer)])
@pytest.mark.asyncio
async def test_ainvoke_pipeline_async_iterator(cls: Tuple[Type, Type]) -> None:
    doc1 = langchain.schema.Document(page_content="My test")
    doc2 = langchain.schema.Document(page_content="Other test")
    runnable = cls[0]() | cls[1]()
    r = await runnable.ainvoke(iter([doc1, doc2]))
    rr = [doc async for doc in r]
    assert rr == [
        langchain.schema.Document(page_content=doc1.page_content.lower()),
        langchain.schema.Document(page_content=doc2.page_content.lower()),
    ]
