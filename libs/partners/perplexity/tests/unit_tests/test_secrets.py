from langchain_perplexity import ChatPerplexity


def test_chat_perplexity_secrets() -> None:
    model = ChatPerplexity(
        model="llama-3.1-sonar-small-128k-online", pplx_api_key="foo"
    )
    assert "foo" not in str(model)
