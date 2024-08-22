import os

from langchain_anthropic import AnthropicLLM

os.environ["ANTHROPIC_API_KEY"] = "foo"


def test_anthropic_model_params() -> None:
    # Test standard tracing params
    llm = AnthropicLLM(model="foo")  # type: ignore[call-arg]

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "anthropic",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 1024,
    }

    llm = AnthropicLLM(model="foo", temperature=0.1)  # type: ignore[call-arg]

    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "anthropic",
        "ls_model_type": "llm",
        "ls_model_name": "foo",
        "ls_max_tokens": 1024,
        "ls_temperature": 0.1,
    }
