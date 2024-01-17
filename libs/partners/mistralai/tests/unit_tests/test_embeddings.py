import os

from langchain_mistralai import MistralAIEmbeddings

os.environ["MISTRAL_API_KEY"] = "foo"


def test_mistral_init() -> None:
    embeddings = MistralAIEmbeddings()
    assert embeddings.model == "mistral-embed"
