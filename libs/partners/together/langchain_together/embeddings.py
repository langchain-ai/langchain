"""Wrapper around Together AI's Embeddings API."""

import logging
import os
import warnings
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import openai
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import (
    BaseModel,
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


class TogetherEmbeddings(BaseModel, Embeddings):
    """Together embedding model integration.

    Setup:
        Install ``langchain_together`` and set environment variable
        ``TOGETHER_API_KEY``.

        .. code-block:: bash

            pip install -U langchain_together
            export TOGETHER_API_KEY="your-api-key"

    Key init args — completion params:
        model: str
            Name of Together model to use.

    Key init args — client params:
      api_key: Optional[SecretStr]

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from __module_name__ import TogetherEmbeddings

            embed = TogetherEmbeddings(
                model="togethercomputer/m2-bert-80M-8k-retrieval",
                # api_key="...",
                # other params...
            )

    Embed single text:
        .. code-block:: python

            input_text = "The meaning of life is 42"
            vector = embed.embed_query(input_text)
            print(vector[:3])

        .. code-block:: python

            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Embed multiple texts:
        .. code-block:: python

             input_texts = ["Document 1...", "Document 2..."]
            vectors = embed.embed_documents(input_texts)
            print(len(vectors))
            # The first 3 coordinates for the first vector
            print(vectors[0][:3])

        .. code-block:: python

            2
            [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]

    Async:
        .. code-block:: python

            vector = await embed.aembed_query(input_text)
           print(vector[:3])

            # multiple:
            # await embed.aembed_documents(input_texts)

        .. code-block:: python

            [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "togethercomputer/m2-bert-80M-8k-retrieval"
    """Embeddings model name to use. 
    Instead, use 'togethercomputer/m2-bert-80M-8k-retrieval' for example.
    """
    dimensions: Optional[int] = None
    """The number of dimensions the resulting output embeddings should have.

    Not yet supported.
    """
    together_api_key: Optional[SecretStr] = Field(default=None, alias="api_key")
    """API Key for Solar API."""
    together_api_base: str = Field(
        default="https://api.together.ai/v1/", alias="base_url"
    )
    """Endpoint URL to use."""
    embedding_ctx_length: int = 4096
    """The maximum number of tokens to embed at once.

    Not yet supported.
    """
    allowed_special: Union[Literal["all"], Set[str]] = set()
    """Not yet supported."""
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    """Not yet supported."""
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch.

    Not yet supported.
    """
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float], Any]] = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to Together embedding API. Can be float, httpx.Timeout or
        None."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding.

    Not yet supported.
    """
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    skip_empty: bool = False
    """Whether to skip empty strings when embedding or raise an error.
    Defaults to not skipping.

    Not yet supported."""
    default_headers: Union[Mapping[str, str], None] = None
    default_query: Union[Mapping[str, object], None] = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    http_client: Union[Any, None] = None
    """Optional httpx.Client. Only used for sync invocations. Must specify
        http_async_client as well if you'd like a custom client for async invocations.
    """
    http_async_client: Union[Any, None] = None
    """Optional httpx.AsyncClient. Only used for async invocations. Must specify
        http_client as well if you'd like a custom client for sync invocations."""

    class Config:
        """Configuration for this pydantic object."""

        extra = "forbid"
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
        together_api_key = get_from_dict_or_env(
            values, "together_api_key", "TOGETHER_API_KEY"
        )
        values["together_api_key"] = (
            convert_to_secret_str(together_api_key) if together_api_key else None
        )
        values["together_api_base"] = values["together_api_base"] or os.getenv(
            "TOGETHER_API_BASE"
        )
        client_params = {
            "api_key": (
                values["together_api_key"].get_secret_value()
                if values["together_api_key"]
                else None
            ),
            "base_url": values["together_api_base"],
            "timeout": values["request_timeout"],
            "max_retries": values["max_retries"],
            "default_headers": values["default_headers"],
            "default_query": values["default_query"],
        }
        if not values.get("client"):
            sync_specific = (
                {"http_client": values["http_client"]} if values["http_client"] else {}
            )
            values["client"] = openai.OpenAI(
                **client_params, **sync_specific
            ).embeddings
        if not values.get("async_client"):
            async_specific = (
                {"http_client": values["http_async_client"]}
                if values["http_async_client"]
                else {}
            )
            values["async_client"] = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).embeddings
        return values

    @property
    def _invocation_params(self) -> Dict[str, Any]:
        params: Dict = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return params

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using passage model.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        params = self._invocation_params
        params["model"] = params["model"]

        for text in texts:
            response = self.client.create(input=text, **params)

            if not isinstance(response, dict):
                response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._invocation_params
        params["model"] = params["model"]

        response = self.client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts using passage model asynchronously.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        params = self._invocation_params
        params["model"] = params["model"]

        for text in texts:
            response = await self.async_client.create(input=text, **params)

            if not isinstance(response, dict):
                response = response.model_dump()
                embeddings.extend([i["embedding"] for i in response["data"]])
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text using query model.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        params = self._invocation_params
        params["model"] = params["model"]

        response = await self.async_client.create(input=text, **params)

        if not isinstance(response, dict):
            response = response.model_dump()
        return response["data"][0]["embedding"]
