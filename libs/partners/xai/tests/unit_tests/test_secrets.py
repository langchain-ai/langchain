from langchain_xai import ChatXAI

MODEL_NAME = "grok-4"


def test_chat_xai_secrets() -> None:
    o = ChatXAI(model=MODEL_NAME, xai_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
