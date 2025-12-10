# libs/test.py

import os
import pytest

pytest.importorskip("langchain_qwq")

from langchain_classic.chat_models import init_chat_model
from langchain_qwq import ChatQwen
from langchain_tests.conftest import CustomPersister, CustomSerializer


def test_init_chat_model_qwen(monkeypatch: pytest.MonkeyPatch) -> None:
    # Set a dummy key so ChatQwen's Pydantic validation passes.
    # We are NOT actually calling the API in this test.
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    llm = init_chat_model(
        model="qwen:Qwen/Qwen2.5-72B-Instruct",
        temperature=0,
    )

    assert isinstance(llm, ChatQwen)
    # Important: do NOT call llm.invoke() here, to avoid network calls.
