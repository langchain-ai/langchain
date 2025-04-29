"""written under MIT Licence, Michael Feil 2023."""

import asyncio
from logging import getLogger
from typing import Any, List, Optional

from langchain_core.embeddings import Embeddings
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import Self

__all__ = ["InfinityEmbeddingsLocal"]

logger = getLogger(__name__)


class InfinityEmbeddingsLocal(BaseModel, Embeddings):
    """Optimized Infinity embedding models.

    https://github.com/michaelfeil/infinity
    This class deploys a local Infinity instance to embed text.
    The class requires async usage.

    Infinity is a class to interact with Embedding Models on https://github.com/michaelfeil/infinity


    Example:
        .. code-block:: python

            from langchain_community.embeddings import InfinityEmbeddingsLocal
            async with InfinityEmbeddingsLocal(
                model="BAAI/bge-small-en-v1.5",
                revision=None,
                device="cpu",
            ) as embedder:
                embeddings = await engine.aembed_documents(["text1", "text2"])
    """

    model: str
    "Underlying model id from huggingface, e.g. BAAI/bge-small-en-v1.5"

    revision: Optional[str] = None
    "Model version, the commit hash from huggingface"

    batch_size: int = 32
    "Internal batch size for inference, e.g. 32"

    device: str = "auto"
    "Device to use for inference, e.g. 'cpu' or 'cuda', or 'mps'"

    backend: str = "torch"
    "Backend for inference, e.g. 'torch' (recommended for ROCm/Nvidia)"
    " or 'optimum' for onnx/tensorrt"

    model_warmup: bool = True
    "Warmup the model with the max batch size."

    engine: Any = None  #: :meta private:
    """Infinity's AsyncEmbeddingEngine."""

    # LLM call kwargs
    model_config = ConfigDict(
        extra="forbid",
        protected_namespaces=(),
    )

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""

        try:
            from infinity_emb import AsyncEmbeddingEngine
        except ImportError:
            raise ImportError(
                "Please install the "
                "`pip install 'infinity_emb[optimum,torch]>=0.0.24'` "
                "package to use the InfinityEmbeddingsLocal."
            )
        self.engine = AsyncEmbeddingEngine(
            model_name_or_path=self.model,
            device=self.device,
            revision=self.revision,
            model_warmup=self.model_warmup,
            batch_size=self.batch_size,
            engine=self.backend,
        )
        return self

    async def __aenter__(self) -> None:
        """start the background worker.
        recommended usage is with the async with statement.

        async with InfinityEmbeddingsLocal(
            model="BAAI/bge-small-en-v1.5",
            revision=None,
            device="cpu",
        ) as embedder:
            embeddings = await engine.aembed_documents(["text1", "text2"])
        """
        await self.engine.__aenter__()

    async def __aexit__(self, *args: Any) -> None:
        """stop the background worker,
        required to free references to the pytorch model."""
        await self.engine.__aexit__(*args)

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Async call out to Infinity's embedding endpoint.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        if not self.engine.running:
            logger.warning(
                "Starting Infinity engine on the fly. This is not recommended."
                "Please start the engine before using it."
            )
            async with self:
                # spawning threadpool for multithreaded encode, tokenization
                embeddings, _ = await self.engine.embed(texts)
            # stopping threadpool on exit
            logger.warning("Stopped infinity engine after usage.")
        else:
            embeddings, _ = await self.engine.embed(texts)
        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        """Async call out to Infinity's embedding endpoint.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        This method is async only.
        """
        logger.warning(
            "This method is async only. "
            "Please use the async version `await aembed_documents`."
        )
        return asyncio.run(self.aembed_documents(texts))

    def embed_query(self, text: str) -> List[float]:
        """ """
        logger.warning(
            "This method is async only."
            " Please use the async version `await aembed_query`."
        )
        return asyncio.run(self.aembed_query(text))
