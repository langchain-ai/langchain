from __future__ import annotations

import logging
import os
import warnings
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)

import numpy as np
import tiktoken
import zhipuai
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
    Extra,
    Field,
    SecretStr,
    root_validator,
)
from langchain_core.utils import (
    convert_to_secret_str,
    get_from_dict_or_env,
    get_pydantic_field_names,
)

logger = logging.getLogger(__name__)


class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """ZhipuAI embedding models.

    To use, you should have the
    environment variable ``OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_glm import ZhipuAIEmbeddings

            zhipuai = ZhipuAIEmbeddings(model=""text_embedding")


    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "embedding-2"
    zhipuai_api_base: Optional[str] = Field(default=None, alias="base_url")
    """Base URL path for API requests, leave blank if not using a proxy or service 
        emulator."""
    zhipuai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    """The maximum number of tokens to embed at once."""
    zhipuai_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""

    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, httpx.Timeout or 
        None."""
    headers: Any = None

    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    http_client: Union[Any, None] = None
    """Optional httpx.Client."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        allow_population_by_field_name = True

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                raise ValueError(f"Found {field_name} supplied twice.")
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            raise ValueError(
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )

        values["model_kwargs"] = extra
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        zhipuai_api_key = get_from_dict_or_env(
            values, "zhipuai_api_key", "ZHIPUAI_API_KEY"
        )
        values["zhipuai_api_key"] = (
            convert_to_secret_str(zhipuai_api_key) if zhipuai_api_key else None
        )
        values["zhipuai_api_base"] = values["zhipuai_api_base"] or os.getenv(
            "OPENAI_API_BASE"
        )
        values["zhipuai_api_type"] = get_from_dict_or_env(
            values,
            "zhipuai_api_type",
            "OPENAI_API_TYPE",
            default="",
        )
        values["zhipuai_proxy"] = get_from_dict_or_env(
            values,
            "zhipuai_proxy",
            "OPENAI_PROXY",
            default="",
        )

        client_params = {
            "api_key": values["zhipuai_api_key"].get_secret_value()
            if values["zhipuai_api_key"]
            else None,
            "base_url": values["zhipuai_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "http_client": values["http_client"],
        }
        if not values.get("client"):
            values["client"] = zhipuai.ZhipuAI(**client_params).embeddings
        return values

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        params: Dict = {"model": self.model, **self.model_kwargs}
        return params

    def _get_len_safe_embeddings(
        self, texts: List[str], *, chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generate length-safe embeddings for a list of texts.
        Args:
            texts (List[str]): A list of texts to embed.
            chunk_size (Optional[int]): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """

        _chunk_size = chunk_size or self.chunk_size

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(texts), _chunk_size))
            except ImportError:
                _iter = range(0, len(texts), _chunk_size)
        else:
            _iter = range(0, len(texts), _chunk_size)

        batched_embeddings: List[List[float]] = []
        for i in _iter:
            response = self.client.create(
                input=texts[i : i + _chunk_size], **self._invocation_params
            )
            if not isinstance(response, dict):
                response = response.dict()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        return batched_embeddings

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        return self._get_len_safe_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        return self.embed_documents([text])[0]
