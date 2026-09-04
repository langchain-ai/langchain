from langchain_openai import AzureOpenAIEmbeddings

AZURE_AD_TOKEN = "token"  # noqa: S105


def test_azure_ad_token_takes_precedence_over_api_key() -> None:
    embeddings = AzureOpenAIEmbeddings(
        api_key="api_key",
        azure_ad_token=AZURE_AD_TOKEN,
        azure_endpoint="https://endpoint.com",
        api_version="2023-05-15",
    )

    assert embeddings.client._client._azure_ad_token == AZURE_AD_TOKEN
