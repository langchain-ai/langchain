from langchain.concise.choice import choice
from langchain.llms.fake import FakeListLLM


def test_choice():
    options = ["red", "orange", "yellow", "green", "blue", "indigo", "violet"]
    result = choice(
        "My favoriate color is what you get when you combine red and yellow?",
        "What is my favorite color?",
        options=options,
        llm=FakeListLLM(["orange"]),
    )
    assert result == "orange"
