"""Callables annotated with names that are only imported for type checking.

Deliberately omits `from __future__ import annotations`, so on Python 3.14+ these
annotations are evaluated lazily and raise `NameError` when evaluated. Importing
this module requires Python 3.14+.
"""

# ruff: noqa: TC004

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel
from typing_extensions import override

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.documents import Document
from langchain_core.language_models import LLM, BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator

    from langchain_core.callbacks import (
        CallbackManagerForLLMRun,
        CallbackManagerForRetrieverRun,
        CallbackManagerForToolRun,
    )
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.messages import BaseMessage
    from langchain_core.runnables import RunnableConfig


def add_one(x: int, config: RunnableConfig) -> int:  # noqa: ARG001
    return x + 1


async def aadd_one(x: int, config: RunnableConfig) -> int:  # noqa: ARG001
    return x + 1


def get_content(message: AIMessage) -> str:
    return message.text


def upper(chunks: Iterator[str]) -> Iterator[str]:
    for chunk in chunks:
        yield chunk.upper()


async def aupper(chunks: AsyncIterator[str]) -> AsyncIterator[str]:
    async for chunk in chunks:
        yield chunk.upper()


_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    return _histories.setdefault(session_id, InMemoryChatMessageHistory())


class FakeChatModel(BaseChatModel):
    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage("hello"))])

    @property
    @override
    def _llm_type(self) -> str:
        return "fake"

    def describe(self, *, config: RunnableConfig) -> list[str]:
        return config.get("tags", [])


class FakeLLM(LLM):
    @override
    def _call(
        self,
        prompt: str,
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> str:
        return "hello"

    @property
    @override
    def _llm_type(self) -> str:
        return "fake"


def define_retriever() -> type[BaseRetriever]:
    # `BaseRetriever` inspects `_get_relevant_documents` on subclass definition.
    class FakeRetriever(BaseRetriever):
        @override
        def _get_relevant_documents(
            self, query: str, *, run_manager: CallbackManagerForRetrieverRun
        ) -> list[Document]:
            return [Document(page_content=query)]

    return FakeRetriever


class EchoInput(BaseModel):
    text: str


class EchoTool(BaseTool):
    name: str = "echo"
    description: str = "Echo the input."
    args_schema: type[BaseModel] = EchoInput

    @override
    def _run(
        self, text: str, run_manager: CallbackManagerForToolRun | None = None
    ) -> str:
        return text
