import os

from langchain_openai import AzureOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "foo"
os.environ["OPENAI_API_VERSION"] = "bar"


def test_openai_model_param() -> None:
    llm = AzureOpenAI(azure_deployment="gpt-35-turbo-instruct", azure_endpoint="baz")

    # Test standard tracing params
    ls_params = llm._get_ls_params()
    assert ls_params == {
        "ls_provider": "azure",
        "ls_model_type": "llm",
        "ls_model_name": "gpt-35-turbo-instruct",
        "ls_temperature": 0.7,
        "ls_max_tokens": 256,
    }
