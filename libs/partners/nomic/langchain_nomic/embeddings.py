import os
from typing import List, Literal, Optional, overload

import nomic  # type: ignore[import]
from langchain_core.embeddings import Embeddings
from nomic import embed


class NomicEmbeddings(Embeddings):
    """NomicEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_nomic import NomicEmbeddings

            model = NomicEmbeddings()
    """

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: Optional[str] = ...,
        dimensionality: Optional[int] = ...,
        inference_mode: Literal["remote"] = ...,
    ):
        ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: Optional[str] = ...,
        dimensionality: Optional[int] = ...,
        inference_mode: Literal["local", "dynamic"],
        device: Optional[str] = ...,
    ):
        ...

    @overload
    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: Optional[str] = ...,
        dimensionality: Optional[int] = ...,
        inference_mode: str,
        device: Optional[str] = ...,
    ):
        ...

    def __init__(
        self,
        *,
        model: str,
        nomic_api_key: Optional[str] = None,
        dimensionality: Optional[int] = None,
        inference_mode: str = "remote",
        device: Optional[str] = None,
        vision_model: Optional[str] = None,
    ):
        """Initialize NomicEmbeddings model.

        Args:
            model: model name
            nomic_api_key: optionally, set the Nomic API key. Uses the NOMIC_API_KEY
                environment variable by default.
            dimensionality: The embedding dimension, for use with Matryoshka-capable
                models. Defaults to full-size.
            inference_mode: How to generate embeddings. One of `remote`, `local`
                (Embed4All), or `dynamic` (automatic). Defaults to `remote`.
            device: The device to use for local embeddings. Choices include
                `cpu`, `gpu`, `nvidia`, `amd`, or a specific device name. See
                the docstring for `GPT4All.__init__` for more info. Typically
                defaults to CPU. Do not use on macOS.
        """
        _api_key = nomic_api_key or os.environ.get("NOMIC_API_KEY")
        if _api_key:
            nomic.login(_api_key)
        self.model = model
        self.dimensionality = dimensionality
        self.inference_mode = inference_mode
        self.device = device
        self.vision_model = vision_model

    def embed(self, texts: List[str], *, task_type: str) -> List[List[float]]:
        """Embed texts.

        Args:
            texts: list of texts to embed
            task_type: the task type to use when embedding. One of `search_query`,
                `search_document`, `classification`, `clustering`
        """

        output = embed.text(
            texts=texts,
            model=self.model,
            task_type=task_type,
            dimensionality=self.dimensionality,
            inference_mode=self.inference_mode,
            device=self.device,
        )
        return output["embeddings"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs.

        Args:
            texts: list of texts to embed as documents
        """
        return self.embed(
            texts=texts,
            task_type="search_document",
        )

    def embed_query(self, text: str) -> List[float]:
        """Embed query text.

        Args:
            text: query text
        """
        return self.embed(
            texts=[text],
            task_type="search_query",
        )[0]

    def embed_image(self, uris: List[str]) -> List[List[float]]:
        return embed.image(
            images=uris,
            model=self.vision_model,
        )["embeddings"]
