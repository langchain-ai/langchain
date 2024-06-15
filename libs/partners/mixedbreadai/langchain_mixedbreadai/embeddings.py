import logging
from typing import List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import Field
from mixedbread_ai.types import EncodingFormat, TruncationStrategy  # type: ignore

from .client import MixedBreadAIClient

logger = logging.getLogger(__name__)


class MixedbreadAIEmbeddings(MixedBreadAIClient, Embeddings):
    """
    Implements the Embeddings interface with Mixedbread AI's text representation
    language models.

    Find out more about us at https://mixedbread.ai

    This implementation uses the Embed API.

    To use this you'll need a Mixedbread AI API key - either pass it to
    api_key parameter or set the MXBAI_API_KEY environment variable.

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

    model: str = "mixedbread-ai/mxbai-embed-large-v1"
    """Model name to use."""
    encoding_format: EncodingFormat = Field(default=EncodingFormat.FLOAT)
    truncation_strategy: TruncationStrategy = Field(default=TruncationStrategy.START)
    normalized: bool = Field(default=True)
    dimensions: Optional[int] = Field(default=None)
    """"The desired number of dimensions in the output vectors. 
    Only applicable for Matryoshka-based models"""
    prompt: Optional[str] = Field(default=None)

    def embed(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings = self._client.embeddings(
            model=self.model,
            input=texts,
            encoding_format=self.encoding_format,
            truncation_strategy=self.truncation_strategy,
            normalized=self.normalized,
            dimensions=self.dimensions,
            prompt=self.prompt,
            request_options=self._request_options,
        ).data
        return [item.embedding for item in embeddings]

    async def aembed(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings = (
            await self._aclient.embeddings(
                model=self.model,
                input=texts,
                encoding_format=self.encoding_format,
                truncation_strategy=self.truncation_strategy,
                normalized=self.normalized,
                dimensions=self.dimensions,
                prompt=self.prompt,
                request_options=self._request_options,
            )
        ).data
        return [item.embedding for item in embeddings]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of document texts.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return self.embed(texts)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Mixedbread AI's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        return await self.aembed(texts)

    def embed_query(self, text: str) -> List[float]:
        """Call out to Mixedbread AI's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return self.embed([text])[0]

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Mixedbread AI's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        return (await self.aembed([text]))[0]
