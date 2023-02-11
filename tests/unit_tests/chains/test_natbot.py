"""Test functionality related to natbot."""

from typing import Any, List, Mapping, Optional

from pydantic import BaseModel

from langchain.chains.natbot.base import NatBotChain
from langchain.llms.base import LLM
from langchain.llms.fake import FakeDictLLM


def test_proper_inputs() -> None:
    """Test that natbot shortens inputs correctly."""
    nat_bot_chain = NatBotChain(llm=FakeDictLLM(), objective="testing")
    url = "foo" * 10000
    browser_content = "foo" * 10000
    output = nat_bot_chain.execute(url, browser_content)
    assert output == "bar"


def test_variable_key_naming() -> None:
    """Test that natbot handles variable key naming correctly."""
    nat_bot_chain = NatBotChain(
        llm=FakeDictLLM(),
        objective="testing",
        input_url_key="u",
        input_browser_content_key="b",
        output_key="c",
    )
    output = nat_bot_chain.execute("foo", "foo")
    assert output == "bar"
