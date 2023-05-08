from langchain.concise.decide import decide
from langchain.llms.fake import FakeListLLM


def test_decide():
    result = decide("2 + 2 = 5", "Is this statement true?", llm=FakeListLLM("NO"))
    assert result == False
