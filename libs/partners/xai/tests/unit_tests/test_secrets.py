from langchain_xai import ChatXAI


def test_chat_xai_secrets() -> None:
    o = ChatXAI(model="grok-beta", xai_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
