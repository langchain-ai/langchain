import os
from typing import cast

from pydantic import SecretStr

from langchain_mistralai import MistralAIEmbeddings

os.environ["MISTRAL_API_KEY"] = "foo"


def test_mistral_init() -> None:
    for model in [
        MistralAIEmbeddings(model="mistral-embed", mistral_api_key="test"),  # type: ignore[call-arg]
        MistralAIEmbeddings(model="mistral-embed", api_key="test"),  # type: ignore[arg-type]
    ]:
        assert model.model == "mistral-embed"
        assert cast(SecretStr, model.mistral_api_key).get_secret_value() == "test"
