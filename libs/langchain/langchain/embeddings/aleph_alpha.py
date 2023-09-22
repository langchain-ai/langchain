import asyncio
from abc import ABC
from types import TracebackType
from typing import Any, Coroutine, Dict, Iterable, List, Optional, Type

from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.schema.embeddings import Embeddings
from langchain.utils import get_from_dict_or_env


class AlephAlphaSemanticEmbeddingAbstractClass(BaseModel, Embeddings, ABC):
    """Aleph Alpha's asymmetric semantic embedding.

    AA provides you with an endpoint to embed a document and a query.
    We offer two types of embeddings-asymetric embedding and symmetric embedding
    To learn more, check out:
    https://docs.aleph-alpha.com/docs/tasks/semantic_embed/
    """

    try:
        from aleph_alpha_client import SemanticRepresentation
    except ImportError:
        raise ValueError(
            "Could not import aleph_alpha_client python package. "
            "Please install it with `pip install aleph_alpha_client`."
        )

    document_representation: SemanticRepresentation
    """Either Document or Symmetric, specified in the child class"""
    query_representation: SemanticRepresentation
    """Either Query or Symmetric, specified in the child class"""

    client: Any  #: :meta private:
    async_client: Any  #: :meta private:

    # Embedding params
    model: str = "luminous-base"
    """Model name to use."""
    compress_to_size: Optional[int] = None
    """Should the returned embeddings come back as an original 5120-dim vector,
    or should it be compressed to 128-dim."""
    normalize: bool = False
    """Should returned embeddings be normalized"""
    contextual_control_threshold: Optional[int] = None
    """Attention control parameters only apply to those tokens that have
    explicitly been set in the request."""
    control_log_additive: bool = True
    """Apply controls on prompt items by adding the log(control_factor)
    to attention scores."""

    # Client params
    aleph_alpha_api_key: Optional[str] = None
    """API key for Aleph Alpha API."""
    host: str = "https://api.aleph-alpha.com"
    """The hostname of the API host.
    The default one is https://api.aleph-alpha.com"""
    hosting: Optional[str] = None
    """Determines in which datacenters the request may be processed.
    You can either set the parameter to "aleph-alpha" or
    omit it (defaulting to None).
    Not setting this value, or setting it to None, gives us maximal flexibility
    in processing your request in our
    own datacenters and on servers hosted with other providers.
    Choose this option for maximal availability.
    Setting it to "aleph-alpha" allows us to only process the request
    in our own datacenters.
    Choose this option for maximal data privacy."""
    request_timeout_seconds: int = 305
    """Client timeout that will be set for HTTP requests in the
    `requests` library's API calls.
    Server will close all requests after 300 seconds with
    an internal server error."""
    total_retries: int = 8
    """The number of retries made in case requests fail with certain retryable
    status codes. If the last
    retry fails a corresponding exception is raised. Note, that between retries
    an exponential backoff
    is applied, starting with 0.5 s after the first retry and doubling for each
    retry made. So with the
    default setting of 8 retries a total wait time of 63.5 s is added between
    the retries."""
    nice: bool = False
    """Setting this to True, will signal to the API that you intend to be
    nice to other users
    by de-prioritizing your request below concurrent ones."""
    concurrency_limit: int = 10
    """The maximum number of simultaneous calls to the API allowed.
    This is set to avoid overwhelming the API."""
    show_progress: bool = False
    """Show progress bar while obtaining the embeddings for documents"""

    async def __aenter__(self) -> "AlephAlphaSemanticEmbeddingAbstractClass":
        """The intended way to use the Async Call is via a context manager.
        Due to that, we only initialize the AsycClient in _aenter,
        and we only close its session in aexit"""
        try:
            from aleph_alpha_client import AsyncClient

            self.async_client = AsyncClient(
                token=self.aleph_alpha_api_key,
                host=self.host,
                hosting=self.hosting,
                request_timeout_seconds=self.request_timeout_seconds,
                total_retries=self.total_retries,
                nice=self.nice,
            )

            await self.async_client.__aenter__()

        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )

        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        await self.async_client.__aexit__(
            exc_type=exc_type, exc_val=exc_val, exc_tb=exc_tb
        )

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict[str, Any]:
        """Validate that api key and python package exists in environment."""
        aleph_alpha_api_key = get_from_dict_or_env(
            values, "aleph_alpha_api_key", "ALEPH_ALPHA_API_KEY"
        )
        try:
            from aleph_alpha_client import Client

            values["client"] = Client(
                token=aleph_alpha_api_key,
                host=values["host"],
                hosting=values["hosting"],
                request_timeout_seconds=values["request_timeout_seconds"],
                total_retries=values["total_retries"],
                nice=values["nice"],
            )

            values["aleph_alpha_api_key"] = aleph_alpha_api_key

        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )

        return values

    @staticmethod
    async def gather_with_concurrency(
        n: int,
        *,
        tasks: Iterable[Coroutine[Any, Any, Any]],
        show_progress: bool = False
    ) -> List[Any]:
        semaphore = asyncio.Semaphore(n)

        async def sem_task(task):
            async with semaphore:
                return await task

        if show_progress:
            try:
                from tqdm.asyncio import tqdm_asyncio

                return await tqdm_asyncio.gather(
                    *(sem_task(task) for task in tasks)
                )

            except ImportError:
                raise ValueError(
                    "Could not import tqdm's async version. "
                    "Please install it with `pip install tqdm`."
                )

        return await asyncio.gather(*(sem_task(task) for task in tasks))

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Make a call to Aleph Alpha's semantics embeddings endpoint.
        We get the document representation for every text.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            from aleph_alpha_client import (
                Prompt,
                SemanticEmbeddingRequest,
            )
        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )

        if self.show_progress:
            try:
                from tqdm import tqdm
            except ImportError:
                raise ValueError(
                    "Could not import tqdm python package. "
                    "Please install it with `pip install tqdm`."
                )

        iterated_texts = tqdm(texts) if self.show_progress else texts

        document_embeddings = [
            self.client.semantic_embed(
                request=SemanticEmbeddingRequest(
                    prompt=Prompt.from_text(text),
                    representation=self.document_representation,
                    compress_to_size=self.compress_to_size,
                    normalize=self.normalize,
                    contextual_control_threshold=\
                        self.contextual_control_threshold,
                    control_log_additive=self.control_log_additive,
                ),
                model=self.model,
            ).embedding
            for text in iterated_texts
        ]

        return document_embeddings

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronously make a call to Aleph Alpha's semantic embeddings
        endpoint. We get the document representation for every text.
        The number of concurrent calls is limited by `concurrency_limit`

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        try:
            from aleph_alpha_client import (
                Prompt,
                SemanticEmbeddingRequest,
            )
        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )

        requests = [
            SemanticEmbeddingRequest(
                prompt=Prompt.from_text(text),
                representation=self.document_representation,
                compress_to_size=self.compress_to_size,
                normalize=self.normalize,
                contextual_control_threshold=self.contextual_control_threshold,
                control_log_additive=self.control_log_additive,
            )
            for text in texts
        ]

        responses = await self.gather_with_concurrency(
            self.concurrency_limit,
            tasks=(
                self.async_client.semantic_embed(request=req, model=self.model)
                for req in requests
            ),
            show_progress=self.show_progress,
        )

        return [r.embedding for r in responses]

    def embed_query(self, text: str) -> List[float]:
        """Make a call to Aleph Alpha's semantic embeddings endpoint.
        We get the query representation for every text.
        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        try:
            from aleph_alpha_client import (
                Prompt,
                SemanticEmbeddingRequest,
            )
        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )
        symmetric_request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(text),
            representation=self.query_representation,
            compress_to_size=self.compress_to_size,
            normalize=self.normalize,
            contextual_control_threshold=self.contextual_control_threshold,
            control_log_additive=self.control_log_additive,
        )

        symmetric_response = self.client.semantic_embed(
            request=symmetric_request, model=self.model
        )

        return symmetric_response.embedding

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronously make a call to Aleph Alpha's
        semantic embeddings endpoint. We get the query
        representation for every text.
        The number of concurrent calls is limited by `concurrency_limit`

        Args:
            text: The query text to embed.

        Returns:
            Embedding for the query.
        """
        try:
            from aleph_alpha_client import (
                Prompt,
                SemanticEmbeddingRequest,
            )
        except ImportError:
            raise ValueError(
                "Could not import aleph_alpha_client python package. "
                "Please install it with `pip install aleph_alpha_client`."
            )

        query_request = SemanticEmbeddingRequest(
            prompt=Prompt.from_text(text),
            representation=self.query_representation,
            compress_to_size=self.compress_to_size,
            normalize=self.normalize,
            contextual_control_threshold=self.contextual_control_threshold,
            control_log_additive=self.control_log_additive,
        )

        query_response = await self.async_client.semantic_embed(
            request=query_request, model=self.model
        )

        return query_response.embedding


class AlephAlphaAsymmetricSemanticEmbedding(
    AlephAlphaSemanticEmbeddingAbstractClass
):
    """
    Example:
        .. code-block:: python
            from aleph_alpha import AlephAlphaAsymmetricSemanticEmbedding

            embeddings = AlephAlphaAsymmetricSemanticEmbedding(
                normalize=True, compress_to_size=128
            )

            document = "This is a content of the document"
            query = "What is the content of the document?"

            doc_result = embeddings.embed_documents([document])
            query_result = embeddings.embed_query(query)

    """

    try:
        from aleph_alpha_client import SemanticRepresentation
    except ImportError:
        raise ValueError(
            "Could not import aleph_alpha_client python package. "
            "Please install it with `pip install aleph_alpha_client`."
        )
    document_representation = SemanticRepresentation.Document
    query_representation = SemanticRepresentation.Query


class AlephAlphaSymmetricSemanticEmbedding(
    AlephAlphaAsymmetricSemanticEmbedding
):
    """
    Example:
        .. code-block:: python

            from aleph_alpha import AlephAlphaSymmetricSemanticEmbedding

            embeddings = AlephAlphaAsymmetricSemanticEmbedding(
                normalize=True, compress_to_size=128
            )
            text = "This is a test text"

            doc_result = embeddings.embed_documents([text])
            query_result = embeddings.embed_query(text)
    """

    try:
        from aleph_alpha_client import SemanticRepresentation
    except ImportError:
        raise ValueError(
            "Could not import aleph_alpha_client python package. "
            "Please install it with `pip install aleph_alpha_client`."
        )
    document_representation = SemanticRepresentation.Symmetric
    query_representation = SemanticRepresentation.Symmetric
