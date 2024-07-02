import logging
from typing import Any, Iterator, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from langchain_core.utils.iter import batch_iterate
from mixedbread_ai.types import EncodingFormat, TruncationStrategy  # type: ignore

from langchain_mixedbreadai.client import MixedBreadAIClient

logger = logging.getLogger(__name__)


class MixedbreadAIEmbeddings(MixedBreadAIClient, Embeddings):
    """
    Implements the Embeddings interface with Mixedbread AI's embedding API.

    Find out more at https://mixedbread.ai/docs

    This implementation uses the embeddings API.

    To use this you'll need a Mixedbread AI API key - either pass it to
    the api_key parameter or set the MXBAI_API_KEY environment variable.

    API keys are available on https://mixedbread.ai - it's free to sign up and trial API
    keys work with this implementation.

    Basic Example:
        .. code-block:: python

            mixedbread_embeddings = MixedbreadAIEmbeddings(
                model="mixedbread-ai/mxbai-embed-large-v1"
            )
            text = "This is a test document."

            query_result = mixedbread_embeddings.embed_query(text)
            print(query_result)

            doc_result = mixedbread_embeddings.embed_documents([text])
            print(doc_result)
    """

    model: str = Field(
        default="mixedbread-ai/mxbai-embed-large-v1",
        description="Model name to use.",
        min_length=1,
    )
    encoding_format: EncodingFormat = Field(
        default=EncodingFormat.FLOAT, description="Encoding format to use."
    )
    truncation_strategy: TruncationStrategy = Field(
        default=TruncationStrategy.START, description="Truncation strategy to use."
    )
    normalized: bool = Field(
        default=True, description="Whether to normalize the embeddings."
    )
    dimensions: Optional[int] = Field(
        default=None,
        description="The desired number of dimensions in the output vectors."
        + " Only applicable for Matryoshka-based models",
    )
    prompt: Optional[str] = Field(
        default=None, description="Prompt to use for the model.", min_length=1
    )
    batch_size: int = Field(
        default=128, description="Batch size for batch processing", ge=1, le=256
    )
    show_progress_bar: bool = Field(
        default=False, description="Show progress of batch processing"
    )

    def _batch_iterate(self, items: List[Any], desc: str) -> Iterator[List[Any]]:
        batch_itr = batch_iterate(self.batch_size, items)
        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e
            batch_itr = tqdm(batch_itr, total=len(items), desc=desc)
        return batch_itr

    def _embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        result: List[List[float]] = []
        for batch in self._batch_iterate(texts, desc="Embedding"):
            embeddings = self._client.embeddings(
                model=self.model,
                input=batch,
                encoding_format=self.encoding_format,
                truncation_strategy=self.truncation_strategy,
                normalized=self.normalized,
                dimensions=self.dimensions,
                prompt=self.prompt,
                request_options=self._request_options,
            ).data
            result.extend([item.embedding for item in embeddings])
        return result

    async def _aembed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        result: List[List[float]] = []
        for batch in self._batch_iterate(texts, desc="Async Embedding"):
            embeddings = (
                await self._aclient.embeddings(
                    model=self.model,
                    input=batch,
                    encoding_format=self.encoding_format,
                    truncation_strategy=self.truncation_strategy,
                    normalized=self.normalized,
                    dimensions=self.dimensions,
                    prompt=self.prompt,
                    request_options=self._request_options,
                )
            ).data
            result.extend([item.embedding for item in embeddings])
        return result

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of document texts using Mixedbread AI's embedding API endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self._embed(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Asynchronously embed a list of document texts
        using Mixedbread AI's embedding API endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return await self._aembed(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        Create embeddings using a call out to Mixedbread AI's embedding API endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self._embed([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """
        Create embeddings using an async call out to
        Mixedbread AI's embedding API endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return (await self._aembed([text]))[0]
