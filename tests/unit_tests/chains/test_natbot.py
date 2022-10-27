"""Test functionality related to natbot."""

from typing import List, Optional

from langchain.chains.natbot.base import NatBotChain
from langchain.llms.base import LLM, CompletionOutput


class FakeLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    def generate(self, prompt: str, stop: Optional[List[str]] = None) -> List[CompletionOutput]:
        """Return `foo` if longer than 10000 words, else `bar`."""
        if len(prompt) > 10000:
            return [CompletionOutput(text="foo")]
        else:
            return [CompletionOutput(text="bar")]


def test_proper_inputs() -> None:
    """Test that natbot shortens inputs correctly."""
    nat_bot_chain = NatBotChain(llm=FakeLLM(), objective="testing")
    url = "foo" * 10000
    browser_content = "foo" * 10000
    output = nat_bot_chain.run(url, browser_content)
    assert output == "bar"


def test_variable_key_naming() -> None:
    """Test that natbot handles variable key naming correctly."""
    nat_bot_chain = NatBotChain(
        llm=FakeLLM(),
        objective="testing",
        input_url_key="u",
        input_browser_content_key="b",
        output_key="c",
    )
    output = nat_bot_chain.run("foo", "foo")
    assert output == "bar"
