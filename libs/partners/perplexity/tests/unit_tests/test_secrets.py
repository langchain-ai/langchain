from langchain_perplexity import ChatPerplexity


def test_chat_perplexity_secrets() -> None:
    o = ChatPerplexity(model="llama-3.1-sonar-small-128k-online", pplx_api_key="foo")  # type: ignore[call-arg]
    s = str(o)
    assert "foo" not in s
