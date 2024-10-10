from __future__ import annotations

import logging
import warnings
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

from langchain_core.embeddings import Embeddings
from langchain_core.utils import (
    get_from_dict_or_env,
    get_pydantic_field_names,
    pre_init,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tenacity import (
    AsyncRetrying,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)


def _create_retry_decorator(embeddings: LocalAIEmbeddings) -> Callable[[Any], Any]:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    return retry(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.APITimeoutError)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
            | retry_if_exception_type(openai.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )


def _async_retry_decorator(embeddings: LocalAIEmbeddings) -> Any:
    import openai

    min_seconds = 4
    max_seconds = 10
    # Wait 2^x * 1 second between each retry starting with
    # 4 seconds, then up to 10 seconds, then 10 seconds afterwards
    async_retrying = AsyncRetrying(
        reraise=True,
        stop=stop_after_attempt(embeddings.max_retries),
        wait=wait_exponential(multiplier=1, min=min_seconds, max=max_seconds),
        retry=(
            retry_if_exception_type(openai.APITimeoutError)
            | retry_if_exception_type(openai.APIError)
            | retry_if_exception_type(openai.APIConnectionError)
            | retry_if_exception_type(openai.RateLimitError)
            | retry_if_exception_type(openai.InternalServerError)
        ),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )

    def wrap(func: Callable) -> Callable:
        async def wrapped_f(*args: Any, **kwargs: Any) -> Callable:
            async for _ in async_retrying:
                return await func(*args, **kwargs)
            raise AssertionError("this is unreachable")

        return wrapped_f

    return wrap


# https://stackoverflow.com/questions/76469415/getting-embeddings-of-length-1-from-langchain-openaiembeddings
def _check_response(response: Any) -> Any:
    if any(len(d.embedding) == 1 for d in response.data):
        import openai

        raise openai.APIError(
            "LocalAI API returned an empty embedding", None, body=None
        )
    return response


def embed_with_retry(embeddings: LocalAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""
    retry_decorator = _create_retry_decorator(embeddings)

    @retry_decorator
    def _embed_with_retry(**kwargs: Any) -> Any:
        response = embeddings.client.create(**kwargs)
        return _check_response(response)

    return _embed_with_retry(**kwargs)


async def async_embed_with_retry(embeddings: LocalAIEmbeddings, **kwargs: Any) -> Any:
    """Use tenacity to retry the embedding call."""

    @_async_retry_decorator(embeddings)
    async def _async_embed_with_retry(**kwargs: Any) -> Any:
        response = await embeddings.async_client.create(**kwargs)
        return _check_response(response)

    return await _async_embed_with_retry(**kwargs)


class LocalAIEmbeddings(BaseModel, Embeddings):
    """LocalAI embedding models.

    Since LocalAI and OpenAI have 1:1 compatibility between APIs, this class
    uses the ``openai`` Python package's ``openai.Embedding`` as its client.
    Thus, you should have the ``openai`` python package installed, and defeat
    the environment variable ``OPENAI_API_KEY`` by setting to a random string.
    You also need to specify ``OPENAI_API_BASE`` to point to your LocalAI
    service endpoint.

    Example:
        .. code-block:: python

            from langchain_community.embeddings import LocalAIEmbeddings
            openai = LocalAIEmbeddings(
                openai_api_key="random-string",
                openai_api_base="http://localhost:8080"
            )
    """

    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: str = "text-embedding-ada-002"
    deployment: str = model
    openai_api_version: Optional[str] = None
    openai_api_base: Optional[str] = None
    # to support explicit proxy for LocalAI
    openai_proxy: Optional[str] = None
    embedding_ctx_length: int = 8191
    """The maximum number of tokens to embed at once."""
    openai_api_key: Optional[str] = None
    openai_organization: Optional[str] = None
    allowed_special: Union[Literal["all"], Set[str]] = set()
    disallowed_special: Union[Literal["all"], Set[str], Sequence[str]] = "all"
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 6
    """Maximum number of retries to make when generating."""
    request_timeout: Optional[Union[float, Tuple[float, float]]] = None
    """Timeout in seconds for the LocalAI request."""
    headers: Any = None
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding."""
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: Dict[str, Any]) -> Any:
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

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["openai_api_key"] = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        values["openai_api_base"] = get_from_dict_or_env(
            values,
            "openai_api_base",
            "OPENAI_API_BASE",
            default="",
        )
        values["openai_proxy"] = get_from_dict_or_env(
            values,
            "openai_proxy",
            "OPENAI_PROXY",
            default="",
        )
        default_api_version = ""
        values["openai_api_version"] = get_from_dict_or_env(
            values,
            "openai_api_version",
            "OPENAI_API_VERSION",
            default=default_api_version,
        )
        values["openai_organization"] = get_from_dict_or_env(
            values,
            "openai_organization",
            "OPENAI_ORGANIZATION",
            default="",
        )
        if values.get("openai_proxy") and (
            values.get("client") or values.get("async_client")
        ):
            raise ValueError(
                "Cannot specify 'openai_proxy' if one of "
                "'client'/'async_client' is already specified. Received:\n"
                f"{values.get('openai_proxy')=}"
            )
        try:
            import openai

            client_params = {
                "api_key": values["openai_api_key"],
                "organization": values["openai_organization"],
                "base_url": values["openai_api_base"],
                "timeout": values["request_timeout"],
                "max_retries": values["max_retries"],
            }
            if not values.get("client"):
                sync_specific = {}
                if values.get("openai_proxy"):
                    try:
                        import httpx
                    except ImportError as e:
                        raise ImportError(
                            "Could not import httpx python package. "
                            "Please install it with `pip install httpx`."
                        ) from e
                    sync_specific["http_client"] = httpx.Client(
                        # httpx>=0.26 like in langchain-openai
                        # proxy=values.get("openai_proxy")
                        # perhaps it might be more restrictive
                        proxies={"all://*": values.get("openai_proxy")}
                    )
                values["client"] = openai.OpenAI(
                    **client_params, **sync_specific
                ).embeddings
            if not values.get("async_client"):
                async_specific = {}
                if values.get("openai_proxy"):
                    try:
                        import httpx
                    except ImportError as e:
                        raise ImportError(
                            "Could not import httpx python package. "
                            "Please install it with `pip install httpx`."
                        ) from e
                    async_specific["http_client"] = httpx.AsyncClient(
                        # httpx>=0.26 like in langchain-openai
                        # proxy=values.get("openai_proxy")
                        # perhaps it might be more restrictive
                        proxies={"all://*": values.get("openai_proxy")}
                    )
                values["async_client"] = openai.AsyncOpenAI(
                    **client_params, **async_specific
                ).embeddings
        except ImportError:
            raise ImportError(
                "Could not import openai python package. "
                "Please install it with `pip install openai`."
            )
        return values

    @property
    def _invocation_params(self) -> Dict:
        openai_args = {
            "model": self.model,
            **self.model_kwargs,
        }
        return openai_args

    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint."""
        # handle large input text
        if self.model.endswith("001"):
            # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
            # replace newlines, which can negatively affect performance.
            text = text.replace("\n", " ")
        return (
            embed_with_retry(
                self,
                input=[text],
                **self._invocation_params,
            )
            .data[0]
            .embedding
        )

    async def _aembedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint."""
        # handle large input text
        if self.model.endswith("001"):
            # See: https://github.com/openai/openai-python/issues/418#issuecomment-1525939500
            # replace newlines, which can negatively affect performance.
            text = text.replace("\n", " ")
        return (
            (
                await async_embed_with_retry(
                    self,
                    input=[text],
                    **self._invocation_params,
                )
            )
            .data[0]
            .embedding
        )

    def embed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to LocalAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        # call _embedding_func for each text
        return [self._embedding_func(text, engine=self.deployment) for text in texts]

    async def aembed_documents(
        self, texts: List[str], chunk_size: Optional[int] = 0
    ) -> List[List[float]]:
        """Call out to LocalAI's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If None, will use the chunk size
                specified by the class.

        Returns:
            List of embeddings, one for each text.
        """
        embeddings = []
        for text in texts:
            response = await self._aembedding_func(text, engine=self.deployment)
            embeddings.append(response)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = self._embedding_func(text, engine=self.deployment)
        return embedding

    async def aembed_query(self, text: str) -> List[float]:
        """Call out to LocalAI's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embedding for the text.
        """
        embedding = await self._aembedding_func(text, engine=self.deployment)
        return embedding
