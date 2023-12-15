"""Wrapper around YandexGPT embedding models."""

from ast import Dict
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env


class YandexGPTEmbeddings(BaseModel, Embeddings):
    """YandexGPT Embeddings models.

    To use, you should have the ``yandexcloud`` python package installed.

    There are two authentication options for the service account
    with the ``ai.languageModels.user`` role:
        - You can specify the token in a constructor parameter `iam_token`
        or in an environment variable `YC_IAM_TOKEN`.
        - You can specify the key in a constructor parameter `api_key`
        or in an environment variable `YC_API_KEY`.

    Example:
        .. code-block:: python

            from langchain_community.embeddings.yandex import YandexGPTEmbeddings
            embeddings = YandexGPTEmbeddings(iam_token="t1.9eu...", model_uri="emb://<folder-id>/text-search-query/latest")
    """

    iam_token: str = ""
    """Yandex Cloud IAM token for service account
    with the `ai.languageModels.user` role"""
    api_key: str = ""
    """Yandex Cloud Api Key for service account
    with the `ai.languageModels.user` role"""
    model_uri: str = ""
    """Model uri to use."""
    url: str = "llm.api.cloud.yandex.net:443"
    """The url of the API."""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that iam token exists in environment."""

        iam_token = get_from_dict_or_env(values, "iam_token", "YC_IAM_TOKEN", "")
        values["iam_token"] = iam_token
        api_key = get_from_dict_or_env(values, "api_key", "YC_API_KEY", "")
        values["api_key"] = api_key
        if api_key == "" and iam_token == "":
            raise ValueError("Either 'YC_API_KEY' or 'YC_IAM_TOKEN' must be provided.")
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a YandexGPT embeddings models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            import grpc
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2 import (  # noqa: E501
                TextEmbeddingRequest,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import (  # noqa: E501
                EmbeddingsServiceStub,
            )
        except ImportError as e:
            raise ImportError(
                "Please install YandexCloud SDK" " with `pip install yandexcloud`."
            ) from e
        result = []
        channel_credentials = grpc.ssl_channel_credentials()
        channel = grpc.secure_channel(self.url, channel_credentials)

        if self.iam_token:
            metadata = (("authorization", f"Bearer {self.iam_token}"),)
        else:
            metadata = (("authorization", f"Api-Key {self.api_key}"),)

        for text in texts:
            request = TextEmbeddingRequest(model_uri=self.model_uri, text=text)
            stub = EmbeddingsServiceStub(channel)
            res = stub.TextEmbedding(request, metadata=metadata)
            result.append(res.embedding)

        return result

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a YandexGPT embeddings models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed_documents([text])[0]
