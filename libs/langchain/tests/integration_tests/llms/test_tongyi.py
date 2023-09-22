"""Test Tongyi API wrapper."""
from typing import Any, Generator

from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms.tongyi import Tongyi
from langchain.schema import LLMResult


class TestCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run when LLM generates a new token."""
        assert isinstance(token, str)


def test_tongyi_call() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    output = llm("who are you")
    assert isinstance(output, str)


def test_tongyi_stream_call() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi(streaming=True, callbacks=[TestCallbackHandler()])
    llm("who are you")


def test_tongyi_generate() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    output = llm.generate(["who are you"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_tongyi_generate_stream() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi(streaming=True)
    output = llm.generate(["who are you"])
    assert isinstance(output, LLMResult)
    assert isinstance(output.generations, list)


def test_tongyi_stream() -> None:
    """Test valid call to tongyi."""
    llm = Tongyi()
    generator = llm.stream("who are you")

    assert isinstance(generator, Generator)

    for token in generator:
        assert isinstance(token, str)
