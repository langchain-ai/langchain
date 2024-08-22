"""Wrapper around YandexGPT embedding models."""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Sequence

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env, pre_init
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


class YandexGPTEmbeddings(BaseModel, Embeddings):
    """YandexGPT Embeddings models.

    To use, you should have the ``yandexcloud`` python package installed.

    There are two authentication options for the service account
    with the ``ai.languageModels.user`` role:
        - You can specify the token in a constructor parameter `iam_token`
        or in an environment variable `YC_IAM_TOKEN`.
        - You can specify the key in a constructor parameter `api_key`
        or in an environment variable `YC_API_KEY`.

    To use the default model specify the folder ID in a parameter `folder_id`
    or in an environment variable `YC_FOLDER_ID`.

    Example:
        .. code-block:: python

            from langchain_community.embeddings.yandex import YandexGPTEmbeddings
            embeddings = YandexGPTEmbeddings(iam_token="t1.9eu...", folder_id=<folder-id>)
    """  # noqa: E501

    iam_token: SecretStr = ""  # type: ignore[assignment]
    """Yandex Cloud IAM token for service account
    with the `ai.languageModels.user` role"""
    api_key: SecretStr = ""  # type: ignore[assignment]
    """Yandex Cloud Api Key for service account
    with the `ai.languageModels.user` role"""
    model_uri: str = Field(default="", alias="query_model_uri")
    """Query model uri to use."""
    doc_model_uri: str = ""
    """Doc model uri to use."""
    folder_id: str = ""
    """Yandex Cloud folder ID"""
    doc_model_name: str = "text-search-doc"
    """Doc model name to use."""
    model_name: str = Field(default="text-search-query", alias="query_model_name")
    """Query model name to use."""
    model_version: str = "latest"
    """Model version to use."""
    url: str = "llm.api.cloud.yandex.net:443"
    """The url of the API."""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    sleep_interval: float = 0.0
    """Delay between API requests"""
    disable_request_logging: bool = False
    """YandexGPT API logs all request data by default. 
    If you provide personal data, confidential information, disable logging."""
    _grpc_metadata: Sequence

    class Config:
        allow_population_by_field_name = True

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that iam token exists in environment."""

        iam_token = convert_to_secret_str(
            get_from_dict_or_env(values, "iam_token", "YC_IAM_TOKEN", "")
        )
        values["iam_token"] = iam_token
        api_key = convert_to_secret_str(
            get_from_dict_or_env(values, "api_key", "YC_API_KEY", "")
        )
        values["api_key"] = api_key
        folder_id = get_from_dict_or_env(values, "folder_id", "YC_FOLDER_ID", "")
        values["folder_id"] = folder_id
        if api_key.get_secret_value() == "" and iam_token.get_secret_value() == "":
            raise ValueError("Either 'YC_API_KEY' or 'YC_IAM_TOKEN' must be provided.")
        if values["iam_token"]:
            values["_grpc_metadata"] = [
                ("authorization", f"Bearer {values['iam_token'].get_secret_value()}")
            ]
            if values["folder_id"]:
                values["_grpc_metadata"].append(("x-folder-id", values["folder_id"]))
        else:
            values["_grpc_metadata"] = (
                ("authorization", f"Api-Key {values['api_key'].get_secret_value()}"),
            )

        if not values.get("doc_model_uri"):
            if values["folder_id"] == "":
                raise ValueError("'doc_model_uri' or 'folder_id' must be provided.")
            values["doc_model_uri"] = (
                f"emb://{values['folder_id']}/{values['doc_model_name']}/{values['model_version']}"
            )
        if not values.get("model_uri"):
            if values["folder_id"] == "":
                raise ValueError("'model_uri' or 'folder_id' must be provided.")
            values["model_uri"] = (
                f"emb://{values['folder_id']}/{values['model_name']}/{values['model_version']}"
            )
        if values["disable_request_logging"]:
            values["_grpc_metadata"].append(
                (
                    "x-data-logging-enabled",
                    "false",
                )
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed documents using a YandexGPT embeddings models.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """

        return _embed_with_retry(self, texts=texts)

    def embed_query(self, text: str) -> List[float]:
        """Embed a query using a YandexGPT embeddings models.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return _embed_with_retry(self, texts=[text], embed_query=True)[0]


def _create_retry_decorator(llm: YandexGPTEmbeddings) -> Callable[[Any], Any]:
    from grpc import RpcError

    min_seconds = 1
    max_seconds = 60
    return retry(
        reraise=True,
        stop=stop_after_attempt(llm.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(retry_if_exception_type((RpcError))),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _embed_with_retry(llm: YandexGPTEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(llm)

    @retry_decorator
    def _completion_with_retry(**_kwargs: Any) -> Any:
        return _make_request(llm, **_kwargs)

    return _completion_with_retry(**kwargs)


def _make_request(self: YandexGPTEmbeddings, texts: List[str], **kwargs):  # type: ignore[no-untyped-def]
    try:
        import grpc

        try:
            from yandex.cloud.ai.foundation_models.v1.embedding.embedding_service_pb2 import (  # noqa: E501
                TextEmbeddingRequest,
            )
            from yandex.cloud.ai.foundation_models.v1.embedding.embedding_service_pb2_grpc import (  # noqa: E501
                EmbeddingsServiceStub,
            )
        except ModuleNotFoundError:
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2 import (  # noqa: E501
                TextEmbeddingRequest,
            )
            from yandex.cloud.ai.foundation_models.v1.foundation_models_service_pb2_grpc import (  # noqa: E501
                EmbeddingsServiceStub,
            )
    except ImportError as e:
        raise ImportError(
            "Please install YandexCloud SDK  with `pip install yandexcloud` \
            or upgrade it to recent version."
        ) from e
    result = []
    channel_credentials = grpc.ssl_channel_credentials()
    channel = grpc.secure_channel(self.url, channel_credentials)
    # Use the query model if embed_query is True
    if kwargs.get("embed_query"):
        model_uri = self.model_uri
    else:
        model_uri = self.doc_model_uri

    for text in texts:
        request = TextEmbeddingRequest(model_uri=model_uri, text=text)
        stub = EmbeddingsServiceStub(channel)
        res = stub.TextEmbedding(request, metadata=self._grpc_metadata)  # type: ignore[attr-defined]
        result.append(list(res.embedding))
        time.sleep(self.sleep_interval)

    return result
