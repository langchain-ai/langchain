"""Test Base Retriever logic."""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest
from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.retrievers import BaseRetriever, Document

from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.fixture
def fake_retriever_v1() -> BaseRetriever:
    with pytest.warns(
        DeprecationWarning,
        match="Retrievers must implement abstract "
        "`_get_relevant_documents` method instead of `get_relevant_documents`",
    ):

        class FakeRetrieverV1(BaseRetriever):
            def get_relevant_documents(  # type: ignore[override]
                self,
                query: str,
            ) -> List[Document]:
                assert isinstance(self, FakeRetrieverV1)
                return [
                    Document(page_content=query, metadata={"uuid": "1234"}),
                ]

            async def aget_relevant_documents(  # type: ignore[override]
                self,
                query: str,
            ) -> List[Document]:
                assert isinstance(self, FakeRetrieverV1)
                return [
                    Document(
                        page_content=f"Async query {query}", metadata={"uuid": "1234"}
                    ),
                ]

        return FakeRetrieverV1()  # type: ignore[abstract]


def test_fake_retriever_v1_upgrade(fake_retriever_v1: BaseRetriever) -> None:
    callbacks = FakeCallbackHandler()
    assert fake_retriever_v1._new_arg_supported is False
    assert fake_retriever_v1._expects_other_args is False
    results: List[Document] = fake_retriever_v1.invoke(
        "Foo", config={"callbacks": [callbacks]}
    )
    assert results[0].page_content == "Foo"
    assert callbacks.retriever_starts == 1
    assert callbacks.retriever_ends == 1
    assert callbacks.retriever_errors == 0


async def test_fake_retriever_v1_upgrade_async(
    fake_retriever_v1: BaseRetriever,
) -> None:
    callbacks = FakeCallbackHandler()
    assert fake_retriever_v1._new_arg_supported is False
    assert fake_retriever_v1._expects_other_args is False
    results: List[Document] = await fake_retriever_v1.ainvoke(
        "Foo", config={"callbacks": [callbacks]}
    )
    assert results[0].page_content == "Async query Foo"
    assert callbacks.retriever_starts == 1
    assert callbacks.retriever_ends == 1
    assert callbacks.retriever_errors == 0


def test_fake_retriever_v1_standard_params(fake_retriever_v1: BaseRetriever) -> None:
    ls_params = fake_retriever_v1._get_ls_params()
    assert ls_params == {"ls_retriever_name": "fakeretrieverv1"}


@pytest.fixture
def fake_retriever_v1_with_kwargs() -> BaseRetriever:
    # Test for things like the Weaviate V1 Retriever.
    with pytest.warns(
        DeprecationWarning,
        match="Retrievers must implement abstract "
        "`_get_relevant_documents` method instead of `get_relevant_documents`",
    ):

        class FakeRetrieverV1(BaseRetriever):
            def get_relevant_documents(  # type: ignore[override]
                self, query: str, where_filter: Optional[Dict[str, object]] = None
            ) -> List[Document]:
                assert isinstance(self, FakeRetrieverV1)
                return [
                    Document(page_content=query, metadata=where_filter or {}),
                ]

            async def aget_relevant_documents(  # type: ignore[override]
                self, query: str, where_filter: Optional[Dict[str, object]] = None
            ) -> List[Document]:
                assert isinstance(self, FakeRetrieverV1)
                return [
                    Document(
                        page_content=f"Async query {query}", metadata=where_filter or {}
                    ),
                ]

        return FakeRetrieverV1()  # type: ignore[abstract]


def test_fake_retriever_v1_with_kwargs_upgrade(
    fake_retriever_v1_with_kwargs: BaseRetriever,
) -> None:
    callbacks = FakeCallbackHandler()
    assert fake_retriever_v1_with_kwargs._new_arg_supported is False
    assert fake_retriever_v1_with_kwargs._expects_other_args is True
    results: List[Document] = fake_retriever_v1_with_kwargs.invoke(
        "Foo", config={"callbacks": [callbacks]}, where_filter={"foo": "bar"}
    )
    assert results[0].page_content == "Foo"
    assert results[0].metadata == {"foo": "bar"}
    assert callbacks.retriever_starts == 1
    assert callbacks.retriever_ends == 1
    assert callbacks.retriever_errors == 0


async def test_fake_retriever_v1_with_kwargs_upgrade_async(
    fake_retriever_v1_with_kwargs: BaseRetriever,
) -> None:
    callbacks = FakeCallbackHandler()
    assert fake_retriever_v1_with_kwargs._new_arg_supported is False
    assert fake_retriever_v1_with_kwargs._expects_other_args is True
    results: List[Document] = await fake_retriever_v1_with_kwargs.ainvoke(
        "Foo", config={"callbacks": [callbacks]}, where_filter={"foo": "bar"}
    )
    assert results[0].page_content == "Async query Foo"
    assert results[0].metadata == {"foo": "bar"}
    assert callbacks.retriever_starts == 1
    assert callbacks.retriever_ends == 1
    assert callbacks.retriever_errors == 0


class FakeRetrieverV2(BaseRetriever):
    throw_error: bool = False

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        assert isinstance(self, FakeRetrieverV2)
        assert run_manager is not None
        assert isinstance(run_manager, CallbackManagerForRetrieverRun)
        if self.throw_error:
            raise ValueError("Test error")
        return [
            Document(page_content=query),
        ]

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[AsyncCallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        assert isinstance(self, FakeRetrieverV2)
        assert run_manager is not None
        assert isinstance(run_manager, AsyncCallbackManagerForRetrieverRun)
        if self.throw_error:
            raise ValueError("Test error")
        return [
            Document(page_content=f"Async query {query}"),
        ]


@pytest.fixture
def fake_retriever_v2() -> BaseRetriever:
    return FakeRetrieverV2()


@pytest.fixture
def fake_erroring_retriever_v2() -> BaseRetriever:
    return FakeRetrieverV2(throw_error=True)


def test_fake_retriever_v2(
    fake_retriever_v2: BaseRetriever, fake_erroring_retriever_v2: BaseRetriever
) -> None:
    callbacks = FakeCallbackHandler()
    assert fake_retriever_v2._new_arg_supported is True
    results = fake_retriever_v2.invoke("Foo", config={"callbacks": [callbacks]})
    assert results[0].page_content == "Foo"
    assert callbacks.retriever_starts == 1
    assert callbacks.retriever_ends == 1
    assert callbacks.retriever_errors == 0
    fake_retriever_v2.invoke("Foo", config={"callbacks": [callbacks]})

    with pytest.raises(ValueError, match="Test error"):
        fake_erroring_retriever_v2.invoke("Foo", config={"callbacks": [callbacks]})
    assert callbacks.retriever_errors == 1


async def test_fake_retriever_v2_async(
    fake_retriever_v2: BaseRetriever, fake_erroring_retriever_v2: BaseRetriever
) -> None:
    callbacks = FakeCallbackHandler()
    assert fake_retriever_v2._new_arg_supported is True
    results = await fake_retriever_v2.ainvoke("Foo", config={"callbacks": [callbacks]})
    assert results[0].page_content == "Async query Foo"
    assert callbacks.retriever_starts == 1
    assert callbacks.retriever_ends == 1
    assert callbacks.retriever_errors == 0
    await fake_retriever_v2.ainvoke("Foo", config={"callbacks": [callbacks]})
    with pytest.raises(ValueError, match="Test error"):
        await fake_erroring_retriever_v2.ainvoke(
            "Foo", config={"callbacks": [callbacks]}
        )


def test_fake_retriever_v2_standard_params(fake_retriever_v2: BaseRetriever) -> None:
    ls_params = fake_retriever_v2._get_ls_params()
    assert ls_params == {"ls_retriever_name": "fakeretrieverv2"}
