"""Unit tests for ReAct."""

from typing import Union

from langchain.agents.react.base import ReActChain, ReActDocstoreAgent
from langchain.agents.tools import Tool
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document
from langchain.schema import AgentAction
from langchain.llms.fake import FakeListLLM

_PAGE_CONTENT = """This is a page about LangChain.

It is a really cool framework.

What isn't there to love about langchain?

Made in 2022."""


class FakeDocstore(Docstore):
    """Fake docstore for testing purposes."""

    def search(self, search: str) -> Union[str, Document]:
        """Return the fake document."""
        document = Document(page_content=_PAGE_CONTENT)
        return document


def test_predict_until_observation_normal() -> None:
    """Test predict_until_observation when observation is made normally."""
    outputs = ["foo\nAction 1: Search[foo]"]
    fake_llm = FakeListLLM(responses=outputs)
    tools = [
        Tool(name="Search", func=lambda x: x, description="foo"),
        Tool(name="Lookup", func=lambda x: x, description="bar"),
    ]
    agent = ReActDocstoreAgent.from_llm_and_tools(fake_llm, tools)
    output = agent.plan([], input="")
    expected_output = AgentAction("Search", "foo", outputs[0])
    assert output == expected_output


def test_predict_until_observation_repeat() -> None:
    """Test when no action is generated initially."""
    outputs = ["foo", " Search[foo]"]
    fake_llm = FakeListLLM(responses=outputs)
    tools = [
        Tool(name="Search", func=lambda x: x, description="foo"),
        Tool(name="Lookup", func=lambda x: x, description="bar"),
    ]
    agent = ReActDocstoreAgent.from_llm_and_tools(fake_llm, tools)
    output = agent.plan([], input="")
    expected_output = AgentAction("Search", "foo", "foo\nAction 1: Search[foo]")
    assert output == expected_output


def test_react_chain() -> None:
    """Test react chain."""
    responses = [
        "I should probably search\nAction 1: Search[langchain]",
        "I should probably lookup\nAction 2: Lookup[made]",
        "Ah okay now I know the answer\nAction 3: Finish[2022]",
    ]
    fake_llm = FakeListLLM(responses=responses)
    react_chain = ReActChain(llm=fake_llm, docstore=FakeDocstore())
    output = react_chain.run("when was langchain made")
    assert output == "2022"


def test_react_chain_bad_action() -> None:
    """Test react chain when bad action given."""
    bad_action_name = "BadAction"
    responses = [
        f"I'm turning evil\nAction 1: {bad_action_name}[langchain]",
        "Oh well\nAction 2: Finish[curses foiled again]",
    ]
    fake_llm = FakeListLLM(responses=responses)
    react_chain = ReActChain(llm=fake_llm, docstore=FakeDocstore())
    output = react_chain.run("when was langchain made")
    assert output == "curses foiled again"
