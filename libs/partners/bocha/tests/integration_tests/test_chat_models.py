import os
import pytest
from langchain_bocha import ChatBocha


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY not set"
)
def test_chat_bocha_invoke():
    llm = ChatBocha(model="deepseek-v4-pro")
    result = llm.invoke("Say hello in one word.")
    assert isinstance(result.content, str)
    assert len(result.content) > 0


@pytest.mark.skipif(
    not os.environ.get("BOCHA_API_KEY"),
    reason="BOCHA_API_KEY not set"
)
def test_chat_bocha_stream():
    llm = ChatBocha(model="deepseek-v4-pro")
    chunks = list(llm.stream("Count to 3."))
    assert len(chunks) > 0
