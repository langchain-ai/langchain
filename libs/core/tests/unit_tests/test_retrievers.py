"""Unit tests for `BaseRetriever` callback lifecycle."""

import asyncio

import pytest

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


class _RetrieverEventCaptureHandler(BaseCallbackHandler):
    """Callback handler that captures retriever lifecycle events."""

    def __init__(self) -> None:
        self.errors: list[BaseException] = []
        self.ends: list[object] = []

    def on_retriever_error(
        self,
        error: BaseException,
        *,
        run_id: object,
        parent_run_id: object | None = None,
        **kwargs: object,
    ) -> None:
        del run_id, parent_run_id, kwargs
        self.errors.append(error)

    def on_retriever_end(
        self,
        output: object,
        *,
        run_id: object,
        parent_run_id: object | None = None,
        **kwargs: object,
    ) -> None:
        del run_id, parent_run_id, kwargs
        self.ends.append(output)


async def test_aretriever_cancellation_reports_error_and_reraises() -> None:
    """Cancelling `ainvoke` must close the run with `on_retriever_error`.

    Regression test: `asyncio.CancelledError` is a `BaseException`, so an
    `except Exception` handler skipped the error callback and left the
    retriever run open.
    """
    handler = _RetrieverEventCaptureHandler()
    entered = asyncio.Event()

    class SlowRetriever(BaseRetriever):
        def _get_relevant_documents(
            self, query: str, *, run_manager: object = None
        ) -> list[Document]:
            del query, run_manager
            return [Document(page_content="x")]

        async def _aget_relevant_documents(
            self, query: str, *, run_manager: object = None
        ) -> list[Document]:
            del query, run_manager
            entered.set()
            await asyncio.sleep(30)
            return [Document(page_content="x")]

    task = asyncio.create_task(
        SlowRetriever().ainvoke("q", config={"callbacks": [handler]})
    )
    await asyncio.wait_for(entered.wait(), timeout=5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(handler.errors) == 1
    assert isinstance(handler.errors[0], asyncio.CancelledError)
    assert handler.ends == []
