import logging
from typing import Iterable, List, Optional

import voyageai  # type: ignore
from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import PrivateAttr
from voyageai import Client

logger = logging.getLogger(__name__)


class VoyageAIEmbeddings(Embeddings):
    """VoyageAIEmbeddings embedding model.

    Example:
        .. code-block:: python

            from langchain_voyageai import VoyageAIEmbeddings

            model = VoyageAIEmbeddings()
    """

    client: Client = PrivateAttr()
    aclient: voyageai.client_async.AsyncClient = PrivateAttr()
    model: str
    batch_size: int
    show_progress_bar: bool = False
    truncation: Optional[bool] = None

    def __init__(
        self,
        model: str,
        voyage_api_key: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
        truncation: Optional[bool] = None,
        show_progress_bar: Optional[bool] = None,
    ):
        self.model = model

        if embed_batch_size is None:
            embed_batch_size = 72 if self.model in ["voyage-2", "voyage-02"] else 7

        self.batch_size = embed_batch_size
        self.client = Client(api_key=voyage_api_key)
        self.aclient = voyageai.client_async.AsyncClient(api_key=voyage_api_key)
        self.truncation = truncation
        self.show_progress_bar = show_progress_bar or self.show_progress_bar

    def _get_batch_iterator(self, texts: List[str]) -> Iterable:
        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            _iter = tqdm(range(0, len(texts), self.batch_size))
        else:
            _iter = range(0, len(texts), self.batch_size)  # type: ignore

        return _iter

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings: List[List[float]] = []

        _iter = self._get_batch_iterator(texts)
        for i in _iter:
            embeddings.extend(
                self.client.embed(
                    texts[i : i + self.batch_size],
                    model=self.model,
                    input_type="document",
                    truncation=self.truncation,
                ).embeddings
            )

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.client.embed(
            [text], model=self.model, input_type="query", truncation=self.truncation
        ).embeddings[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []

        _iter = self._get_batch_iterator(texts)
        for i in _iter:
            r = await self.aclient.embed(
                texts[i : i + self.batch_size],
                model=self.model,
                input_type="document",
                truncation=self.truncation,
            )
            embeddings.extend(r.embeddings)

        return embeddings

    async def aembed_query(self, text: str) -> List[float]:
        r = await self.aclient.embed(
            [text],
            model=self.model,
            input_type="query",
            truncation=self.truncation,
        )
        return r.embeddings[0]
