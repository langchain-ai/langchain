"""Public APIs accept callables annotated with names imported under `TYPE_CHECKING`."""

import inspect
import sys
from typing import TYPE_CHECKING

import pytest

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import RunnableGenerator, RunnableLambda
from langchain_core.runnables.history import RunnableWithMessageHistory

pytestmark = pytest.mark.skipif(
    sys.version_info < (3, 14), reason="Requires lazily evaluated annotations."
)

if TYPE_CHECKING or sys.version_info >= (3, 14):
    from tests.unit_tests.utils import _type_checking_annotations as annotated


def test_annotations_are_unresolvable() -> None:
    with pytest.raises(NameError):
        inspect.signature(annotated.add_one)


def test_runnable_lambda() -> None:
    runnable = RunnableLambda(annotated.add_one)
    assert runnable.invoke(1) == 2
    assert runnable.batch([1, 2]) == [2, 3]
    assert list(runnable.stream(1)) == [2]


async def test_runnable_lambda_async() -> None:
    runnable = RunnableLambda(annotated.add_one, afunc=annotated.aadd_one)
    assert await runnable.ainvoke(1) == 2


def test_runnable_generator() -> None:
    runnable = RunnableGenerator(annotated.upper)
    assert "".join(runnable.stream("ab")) == "AB"


async def test_runnable_generator_async() -> None:
    runnable = RunnableGenerator(annotated.upper, annotated.aupper)
    assert "".join([chunk async for chunk in runnable.astream("ab")]) == "AB"


def test_chat_model() -> None:
    model = annotated.FakeChatModel()
    assert model.invoke("hi").content == "hello"
    chain = model | RunnableLambda(annotated.get_content)
    assert chain.invoke("hi") == "hello"


def test_runnable_binding_injects_config() -> None:
    model = annotated.FakeChatModel().with_config(tags=["a"])
    assert model.describe() == ["a"]  # type: ignore[attr-defined]


def test_llm() -> None:
    assert annotated.FakeLLM().invoke("hi") == "hello"


def test_retriever() -> None:
    docs = annotated.define_retriever()().invoke("query")
    assert [doc.page_content for doc in docs] == ["query"]


def test_tool() -> None:
    assert annotated.EchoTool().invoke({"text": "hi"}) == "hi"


@pytest.mark.filterwarnings(
    "ignore::langchain_core._api.deprecation.LangChainDeprecationWarning"
)
def test_runnable_with_message_history() -> None:
    def respond(_: list[BaseMessage]) -> AIMessage:
        return AIMessage("hello")

    runnable = RunnableWithMessageHistory(
        RunnableLambda(respond), annotated.get_session_history
    )
    runnable.invoke([HumanMessage("hi")], config={"configurable": {"session_id": "1"}})
    assert len(annotated.get_session_history("1").messages) == 2
