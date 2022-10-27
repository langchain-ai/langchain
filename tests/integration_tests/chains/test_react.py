"""Integration test for self ask with search."""

from langchain.chains.react.base import ReActChain
from langchain.llms.openai import OpenAI


def test_react() -> None:
    """Test functionality on a prompt."""
    llm = OpenAI(temperature=0)
    react = ReActChain(llm=llm)
    question = "Were Scott Derrickson and Ed Wood of the same nationality?"
    output = react.run(question)
    assert output == "yes"
