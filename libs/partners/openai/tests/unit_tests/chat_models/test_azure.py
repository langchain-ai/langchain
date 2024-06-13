"""Test Azure OpenAI Chat API wrapper."""

from langchain_openai import AzureChatOpenAI


def test_initialize_azure_openai() -> None:
    llm = AzureChatOpenAI(
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        azure_endpoint="my-base-url",
    )
    assert llm.deployment_name == "35-turbo-dev"
    assert llm.openai_api_version == "2023-05-15"
    assert llm.azure_endpoint == "my-base-url"


def test_initialize_more() -> None:
    llm = AzureChatOpenAI(
        api_key="xyz",
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        temperature=0,
    )
    assert llm.openai_api_key is not None
    assert llm.openai_api_key.get_secret_value() == "xyz"
    assert llm.azure_endpoint == "my-base-url"
    assert llm.deployment_name == "35-turbo-dev"
    assert llm.openai_api_version == "2023-05-15"
    assert llm.temperature == 0

    ls_params = llm._get_ls_params()
    assert ls_params["ls_provider"] == "azure"
    assert ls_params["ls_model_name"] == "35-turbo-dev"
