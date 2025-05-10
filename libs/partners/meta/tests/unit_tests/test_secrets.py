from langchain_meta import ChatLlama


def test_chat_llama_secrets() -> None:
    o = ChatLlama(model="Llama-3.3-8B-Instruct", llama_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
