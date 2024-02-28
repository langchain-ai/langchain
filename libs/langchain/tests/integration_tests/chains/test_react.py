"""Integration test for self ask with search."""

from langchain_community.llms.openai import OpenAI

from langchain.agents.react.base import ReActChain
from langchain.docstore.wikipedia import Wikipedia


def test_react() -> None:
    """Test functionality on a prompt."""
    llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo-instruct")
    react = ReActChain(llm=llm, docstore=Wikipedia())
    question = (
        "Author David Chanoff has collaborated with a U.S. Navy admiral "
        "who served as the ambassador to the United Kingdom under "
        "which President?"
    )
    output = react.run(question)
    assert output == "Bill Clinton"
