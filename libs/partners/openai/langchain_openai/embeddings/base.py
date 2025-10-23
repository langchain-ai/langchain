"""Base classes for OpenAI embeddings."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from typing import Any, Literal, cast

import openai
import tiktoken
from langchain_core.embeddings import Embeddings
from langchain_core.runnables.config import run_in_executor
from langchain_core.utils import from_env, get_pydantic_field_names, secret_from_env
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_openai.chat_models._client_utils import _resolve_sync_and_async_api_keys

logger = logging.getLogger(__name__)


def _process_batched_chunked_embeddings(
    num_texts: int,
    tokens: list[list[int] | str],
    batched_embeddings: list[list[float]],
    indices: list[int],
    skip_empty: bool,
) -> list[list[float] | None]:
    # for each text, this is the list of embeddings (list of list of floats)
    # corresponding to the chunks of the text
    results: list[list[list[float]]] = [[] for _ in range(num_texts)]

    # for each text, this is the token length of each chunk
    # for transformers tokenization, this is the string length
    # for tiktoken, this is the number of tokens
    num_tokens_in_batch: list[list[int]] = [[] for _ in range(num_texts)]

    for i in range(len(indices)):
        if skip_empty and len(batched_embeddings[i]) == 1:
            continue
        results[indices[i]].append(batched_embeddings[i])
        num_tokens_in_batch[indices[i]].append(len(tokens[i]))

    # for each text, this is the final embedding
    embeddings: list[list[float] | None] = []
    for i in range(num_texts):
        # an embedding for each chunk
        _result: list[list[float]] = results[i]

        if len(_result) == 0:
            # this will be populated with the embedding of an empty string
            # in the sync or async code calling this
            embeddings.append(None)
            continue

        if len(_result) == 1:
            # if only one embedding was produced, use it
            embeddings.append(_result[0])
            continue

        # else we need to weighted average
        # should be same as
        # average = np.average(_result, axis=0, weights=num_tokens_in_batch[i])
        total_weight = sum(num_tokens_in_batch[i])
        average = [
            sum(
                val * weight
                for val, weight in zip(embedding, num_tokens_in_batch[i], strict=False)
            )
            / total_weight
            for embedding in zip(*_result, strict=False)
        ]

        # should be same as
        # embeddings.append((average / np.linalg.norm(average)).tolist())
        magnitude = sum(val**2 for val in average) ** 0.5
        embeddings.append([val / magnitude for val in average])

    return embeddings


class OpenAIEmbeddings(BaseModel, Embeddings):
    """OpenAI embedding model integration.

    Setup:
        Install `langchain_openai` and set environment variable `OPENAI_API_KEY`.

        ```bash
        pip install -U langchain_openai
        export OPENAI_API_KEY="your-api-key"
        ```

    Key init args — embedding params:
        model:
            Name of OpenAI model to use.
        dimensions:
            The number of dimensions the resulting output embeddings should have.
            Only supported in `'text-embedding-3'` and later models.

    Key init args — client params:
        api_key:
            OpenAI API key.
        organization:
            OpenAI organization ID. If not passed in will be read
            from env var `OPENAI_ORG_ID`.
        max_retries:
            Maximum number of retries to make when generating.
        request_timeout:
            Timeout for requests to OpenAI completion API

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        ```python
        from langchain_openai import OpenAIEmbeddings

        embed = OpenAIEmbeddings(
            model="text-embedding-3-large"
            # With the `text-embedding-3` class
            # of models, you can specify the size
            # of the embeddings you want returned.
            # dimensions=1024
        )
        ```

    Embed single text:
        ```python
        input_text = "The meaning of life is 42"
        vector = embeddings.embed_query("hello")
        print(vector[:3])
        ```
        ```python
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    Embed multiple texts:
        ```python
        vectors = embeddings.embed_documents(["hello", "goodbye"])
        # Showing only the first 3 coordinates
        print(len(vectors))
        print(vectors[0][:3])
        ```
        ```python
        2
        [-0.024603435769677162, -0.007543657906353474, 0.0039630369283258915]
        ```

    Async:
        ```python
        await embed.aembed_query(input_text)
        print(vector[:3])

        # multiple:
        # await embed.aembed_documents(input_texts)
        ```
        ```python
        [-0.009100092574954033, 0.005071679595857859, -0.0029193938244134188]
        ```
    """

    client: Any = Field(default=None, exclude=True)  #: :meta private:
    async_client: Any = Field(default=None, exclude=True)  #: :meta private:
    model: str = "text-embedding-ada-002"
    dimensions: int | None = None
    """The number of dimensions the resulting output embeddings should have.

    Only supported in `text-embedding-3` and later models.
    """
    # to support Azure OpenAI Service custom deployment names
    deployment: str | None = model
    # TODO: Move to AzureOpenAIEmbeddings.
    openai_api_version: str | None = Field(
        default_factory=from_env("OPENAI_API_VERSION", default=None),
        alias="api_version",
    )
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    # to support Azure OpenAI Service custom endpoints
    openai_api_base: str | None = Field(
        alias="base_url", default_factory=from_env("OPENAI_API_BASE", default=None)
    )
    """Base URL path for API requests, leave blank if not using a proxy or service
        emulator."""
    # to support Azure OpenAI Service custom endpoints
    openai_api_type: str | None = Field(
        default_factory=from_env("OPENAI_API_TYPE", default=None)
    )
    # to support explicit proxy for OpenAI
    openai_proxy: str | None = Field(
        default_factory=from_env("OPENAI_PROXY", default=None)
    )
    embedding_ctx_length: int = 8191
    """The maximum number of tokens to embed at once."""
    openai_api_key: (
        SecretStr | None | Callable[[], str] | Callable[[], Awaitable[str]]
    ) = Field(
        alias="api_key", default_factory=secret_from_env("OPENAI_API_KEY", default=None)
    )
    """Automatically inferred from env var `OPENAI_API_KEY` if not provided."""
    openai_organization: str | None = Field(
        alias="organization",
        default_factory=from_env(
            ["OPENAI_ORG_ID", "OPENAI_ORGANIZATION"], default=None
        ),
    )
    """Automatically inferred from env var `OPENAI_ORG_ID` if not provided."""
    allowed_special: Literal["all"] | set[str] | None = None
    disallowed_special: Literal["all"] | set[str] | Sequence[str] | None = None
    chunk_size: int = 1000
    """Maximum number of texts to embed in each batch"""
    max_retries: int = 2
    """Maximum number of retries to make when generating."""
    request_timeout: float | tuple[float, float] | Any | None = Field(
        default=None, alias="timeout"
    )
    """Timeout for requests to OpenAI completion API. Can be float, `httpx.Timeout` or
    None."""
    headers: Any = None
    tiktoken_enabled: bool = True
    """Set this to False for non-OpenAI implementations of the embeddings API, e.g.
    the `--extensions openai` extension for `text-generation-webui`"""
    tiktoken_model_name: str | None = None
    """The model name to pass to tiktoken when using this class.
    Tiktoken is used to count the number of tokens in documents to constrain
    them to be under a certain limit. By default, when set to None, this will
    be the same as the embedding model name. However, there are some cases
    where you may want to use this Embedding class with a model name not
    supported by tiktoken. This can include when using Azure embeddings or
    when using one of the many model providers that expose an OpenAI-like
    API but with different models. In those cases, in order to avoid erroring
    when tiktoken is called, you can specify a model name to use here."""
    show_progress_bar: bool = False
    """Whether to show a progress bar when embedding."""
    model_kwargs: dict[str, Any] = Field(default_factory=dict)
    """Holds any model parameters valid for `create` call not explicitly specified."""
    skip_empty: bool = False
    """Whether to skip empty strings when embedding or raise an error."""
    default_headers: Mapping[str, str] | None = None
    default_query: Mapping[str, object] | None = None
    # Configure a custom httpx client. See the
    # [httpx documentation](https://www.python-httpx.org/api/#client) for more details.
    retry_min_seconds: int = 4
    """Min number of seconds to wait between retries"""
    retry_max_seconds: int = 20
    """Max number of seconds to wait between retries"""
    http_client: Any | None = None
    """Optional `httpx.Client`. Only used for sync invocations. Must specify
        `http_async_client` as well if you'd like a custom client for async
        invocations.
    """
    http_async_client: Any | None = None
    """Optional `httpx.AsyncClient`. Only used for async invocations. Must specify
        `http_client` as well if you'd like a custom client for sync invocations."""
    check_embedding_ctx_length: bool = True
    """Whether to check the token length of inputs and automatically split inputs
        longer than embedding_ctx_length."""

    model_config = ConfigDict(
        extra="forbid", populate_by_name=True, protected_namespaces=()
    )

    @model_validator(mode="before")
    @classmethod
    def build_extra(cls, values: dict[str, Any]) -> Any:
        """Build extra kwargs from additional params that were passed in."""
        all_required_field_names = get_pydantic_field_names(cls)
        extra = values.get("model_kwargs", {})
        for field_name in list(values):
            if field_name in extra:
                msg = f"Found {field_name} supplied twice."
                raise ValueError(msg)
            if field_name not in all_required_field_names:
                warnings.warn(
                    f"""WARNING! {field_name} is not default parameter.
                    {field_name} was transferred to model_kwargs.
                    Please confirm that {field_name} is what you intended."""
                )
                extra[field_name] = values.pop(field_name)

        invalid_model_kwargs = all_required_field_names.intersection(extra.keys())
        if invalid_model_kwargs:
            msg = (
                f"Parameters {invalid_model_kwargs} should be specified explicitly. "
                f"Instead they were passed in as part of `model_kwargs` parameter."
            )
            raise ValueError(msg)

        values["model_kwargs"] = extra
        return values

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.openai_api_type in ("azure", "azure_ad", "azuread"):
            msg = (
                "If you are using Azure, please use the `AzureOpenAIEmbeddings` class."
            )
            raise ValueError(msg)

        # Resolve API key from SecretStr or Callable
        sync_api_key_value: str | Callable[[], str] | None = None
        async_api_key_value: str | Callable[[], Awaitable[str]] | None = None

        if self.openai_api_key is not None:
            # Because OpenAI and AsyncOpenAI clients support either sync or async
            # callables for the API key, we need to resolve separate values here.
            sync_api_key_value, async_api_key_value = _resolve_sync_and_async_api_keys(
                self.openai_api_key
            )

        client_params: dict = {
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        if self.openai_proxy and (self.http_client or self.http_async_client):
            openai_proxy = self.openai_proxy
            http_client = self.http_client
            http_async_client = self.http_async_client
            msg = (
                "Cannot specify 'openai_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{openai_proxy=}\n{http_client=}\n{http_async_client=}"
            )
            raise ValueError(msg)
        if not self.client:
            if sync_api_key_value is None:
                # No valid sync API key, leave client as None and raise informative
                # error on invocation.
                self.client = None
            else:
                if self.openai_proxy and not self.http_client:
                    try:
                        import httpx
                    except ImportError as e:
                        msg = (
                            "Could not import httpx python package. "
                            "Please install it with `pip install httpx`."
                        )
                        raise ImportError(msg) from e
                    self.http_client = httpx.Client(proxy=self.openai_proxy)
                sync_specific = {
                    "http_client": self.http_client,
                    "api_key": sync_api_key_value,
                }
                self.client = openai.OpenAI(**client_params, **sync_specific).embeddings  # type: ignore[arg-type]
        if not self.async_client:
            if self.openai_proxy and not self.http_async_client:
                try:
                    import httpx
                except ImportError as e:
                    msg = (
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    )
                    raise ImportError(msg) from e
                self.http_async_client = httpx.AsyncClient(proxy=self.openai_proxy)
            async_specific = {
                "http_client": self.http_async_client,
                "api_key": async_api_key_value,
            }
            self.async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,  # type: ignore[arg-type]
            ).embeddings
        return self

    @property
    def _invocation_params(self) -> dict[str, Any]:
        params: dict = {"model": self.model, **self.model_kwargs}
        if self.dimensions is not None:
            params["dimensions"] = self.dimensions
        return params

    def _ensure_sync_client_available(self) -> None:
        """Check that sync client is available, raise error if not."""
        if self.client is None:
            msg = (
                "Sync client is not available. This happens when an async callable "
                "was provided for the API key. Use async methods (ainvoke, astream) "
                "instead, or provide a string or sync callable for the API key."
            )
            raise ValueError(msg)

    def _tokenize(
        self, texts: list[str], chunk_size: int
    ) -> tuple[Iterable[int], list[list[int] | str], list[int]]:
        """Take the input `texts` and `chunk_size` and return 3 iterables as a tuple.

        We have `batches`, where batches are sets of individual texts
        we want responses from the openai api. The length of a single batch is
        `chunk_size` texts.

        Each individual text is also split into multiple texts based on the
        `embedding_ctx_length` parameter (based on number of tokens).

        This function returns a 3-tuple of the following:

        _iter: An iterable of the starting index in `tokens` for each *batch*
        tokens: A list of tokenized texts, where each text has already been split
            into sub-texts based on the `embedding_ctx_length` parameter. In the
            case of tiktoken, this is a list of token arrays. In the case of
            HuggingFace transformers, this is a list of strings.
        indices: An iterable of the same length as `tokens` that maps each token-array
            to the index of the original text in `texts`.
        """
        tokens: list[list[int] | str] = []
        indices: list[int] = []
        model_name = self.tiktoken_model_name or self.model

        # If tiktoken flag set to False
        if not self.tiktoken_enabled:
            try:
                from transformers import AutoTokenizer
            except ImportError:
                msg = (
                    "Could not import transformers python package. "
                    "This is needed for OpenAIEmbeddings to work without "
                    "`tiktoken`. Please install it with `pip install transformers`. "
                )
                raise ValueError(msg)

            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=model_name
            )
            for i, text in enumerate(texts):
                # Tokenize the text using HuggingFace transformers
                tokenized: list[int] = tokenizer.encode(text, add_special_tokens=False)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(tokenized), self.embedding_ctx_length):
                    token_chunk: list[int] = tokenized[
                        j : j + self.embedding_ctx_length
                    ]

                    # Convert token IDs back to a string
                    chunk_text: str = tokenizer.decode(token_chunk)
                    tokens.append(chunk_text)
                    indices.append(i)
        else:
            try:
                encoding = tiktoken.encoding_for_model(model_name)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            encoder_kwargs: dict[str, Any] = {
                k: v
                for k, v in {
                    "allowed_special": self.allowed_special,
                    "disallowed_special": self.disallowed_special,
                }.items()
                if v is not None
            }
            for i, text in enumerate(texts):
                if self.model.endswith("001"):
                    # See: https://github.com/openai/openai-python/
                    #      issues/418#issuecomment-1525939500
                    # replace newlines, which can negatively affect performance.
                    text = text.replace("\n", " ")

                if encoder_kwargs:
                    token = encoding.encode(text, **encoder_kwargs)
                else:
                    token = encoding.encode_ordinary(text)

                # Split tokens into chunks respecting the embedding_ctx_length
                for j in range(0, len(token), self.embedding_ctx_length):
                    tokens.append(token[j : j + self.embedding_ctx_length])
                    indices.append(i)

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm

                _iter: Iterable = tqdm(range(0, len(tokens), chunk_size))
            except ImportError:
                _iter = range(0, len(tokens), chunk_size)
        else:
            _iter = range(0, len(tokens), chunk_size)
        return _iter, tokens, indices

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    def _get_len_safe_embeddings(
        self,
        texts: list[str],
        *,
        engine: str,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Generate length-safe embeddings for a list of texts.

        This method handles tokenization and embedding generation, respecting the
        set embedding context length and chunk size. It supports both tiktoken
        and HuggingFace tokenizer based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (int | None): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """
        _chunk_size = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        _iter, tokens, indices = self._tokenize(texts, _chunk_size)
        batched_embeddings: list[list[float]] = []
        for i in _iter:
            response = self.client.create(
                input=tokens[i : i + _chunk_size], **client_kwargs
            )
            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        embeddings = _process_batched_chunked_embeddings(
            len(texts), tokens, batched_embeddings, indices, self.skip_empty
        )
        _cached_empty_embedding: list[float] | None = None

        def empty_embedding() -> list[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = self.client.create(input="", **client_kwargs)
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else empty_embedding() for e in embeddings]

    # please refer to
    # https://github.com/openai/openai-cookbook/blob/main/examples/Embedding_long_inputs.ipynb
    async def _aget_len_safe_embeddings(
        self,
        texts: list[str],
        *,
        engine: str,
        chunk_size: int | None = None,
        **kwargs: Any,
    ) -> list[list[float]]:
        """Asynchronously generate length-safe embeddings for a list of texts.

        This method handles tokenization and asynchronous embedding generation,
        respecting the set embedding context length and chunk size. It supports both
        `tiktoken` and HuggingFace `tokenizer` based on the tiktoken_enabled flag.

        Args:
            texts (List[str]): A list of texts to embed.
            engine (str): The engine or model to use for embeddings.
            chunk_size (int | None): The size of chunks for processing embeddings.

        Returns:
            List[List[float]]: A list of embeddings for each input text.
        """
        _chunk_size = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        _iter, tokens, indices = await run_in_executor(
            None, self._tokenize, texts, _chunk_size
        )
        batched_embeddings: list[list[float]] = []
        for i in range(0, len(tokens), _chunk_size):
            response = await self.async_client.create(
                input=tokens[i : i + _chunk_size], **client_kwargs
            )

            if not isinstance(response, dict):
                response = response.model_dump()
            batched_embeddings.extend(r["embedding"] for r in response["data"])

        embeddings = _process_batched_chunked_embeddings(
            len(texts), tokens, batched_embeddings, indices, self.skip_empty
        )
        _cached_empty_embedding: list[float] | None = None

        async def empty_embedding() -> list[float]:
            nonlocal _cached_empty_embedding
            if _cached_empty_embedding is None:
                average_embedded = await self.async_client.create(
                    input="", **client_kwargs
                )
                if not isinstance(average_embedded, dict):
                    average_embedded = average_embedded.model_dump()
                _cached_empty_embedding = average_embedded["data"][0]["embedding"]
            return _cached_empty_embedding

        return [e if e is not None else await empty_embedding() for e in embeddings]

    def embed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If `None`, will use the chunk size
                specified by the class.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each text.
        """
        self._ensure_sync_client_available()
        chunk_size_ = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        if not self.check_embedding_ctx_length:
            embeddings: list[list[float]] = []
            for i in range(0, len(texts), chunk_size_):
                response = self.client.create(
                    input=texts[i : i + chunk_size_], **client_kwargs
                )
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return self._get_len_safe_embeddings(
            texts, engine=engine, chunk_size=chunk_size, **kwargs
        )

    async def aembed_documents(
        self, texts: list[str], chunk_size: int | None = None, **kwargs: Any
    ) -> list[list[float]]:
        """Call out to OpenAI's embedding endpoint async for embedding search docs.

        Args:
            texts: The list of texts to embed.
            chunk_size: The chunk size of embeddings. If `None`, will use the chunk size
                specified by the class.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            List of embeddings, one for each text.
        """
        chunk_size_ = chunk_size or self.chunk_size
        client_kwargs = {**self._invocation_params, **kwargs}
        if not self.check_embedding_ctx_length:
            embeddings: list[list[float]] = []
            for i in range(0, len(texts), chunk_size_):
                response = await self.async_client.create(
                    input=texts[i : i + chunk_size_], **client_kwargs
                )
                if not isinstance(response, dict):
                    response = response.model_dump()
                embeddings.extend(r["embedding"] for r in response["data"])
            return embeddings

        # NOTE: to keep things simple, we assume the list may contain texts longer
        #       than the maximum context and use length-safe embedding function.
        engine = cast(str, self.deployment)
        return await self._aget_len_safe_embeddings(
            texts, engine=engine, chunk_size=chunk_size, **kwargs
        )

    def embed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            Embedding for the text.
        """
        self._ensure_sync_client_available()
        return self.embed_documents([text], **kwargs)[0]

    async def aembed_query(self, text: str, **kwargs: Any) -> list[float]:
        """Call out to OpenAI's embedding endpoint async for embedding query text.

        Args:
            text: The text to embed.
            kwargs: Additional keyword arguments to pass to the embedding API.

        Returns:
            Embedding for the text.
        """
        embeddings = await self.aembed_documents([text], **kwargs)
        return embeddings[0]
