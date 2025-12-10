import pytest

from langchain.chat_models import init_chat_model

langchain_qwq = pytest.importorskip("langchain_qwq")


def test_init_chat_model_qwen(monkeypatch) -> None:
    """init_chat_model returns ChatQwen when the qwen provider is used."""
    from langchain_qwq import ChatQwen

    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    llm = init_chat_model(
        model="qwen:Qwen/Qwen2.5-72B-Instruct",
        temperature=0,
    )

    assert isinstance(llm, ChatQwen)
