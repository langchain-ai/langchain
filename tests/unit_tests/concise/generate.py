from langchain.concise.generate import generate
from langchain.llms.fake import FakeListLLM


def test_generate():
    result = generate("What is the capital of France?", llm=FakeListLLM(["Paris"]))
    assert isinstance(result, str)
    assert len(result) > 0
