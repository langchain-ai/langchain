import os

from langchain_openai import AzureOpenAIEmbeddings


def test_initialize_azure_openai() -> None:
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-large",
        api_key="xyz",
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
    )
    assert embeddings.model == "text-embedding-large"


def test_intialize_azure_openai_with_base_set() -> None:
    os.environ["OPENAI_API_BASE"] = "https://api.openai.com"
    embeddings = AzureOpenAIEmbeddings(
        model="text-embedding-large",
        api_key="xyz",
        azure_endpoint="my-base-url",
        azure_deployment="35-turbo-dev",
        openai_api_version="2023-05-15",
        openai_api_base=None,
    )
    assert embeddings.model == "text-embedding-large"
