import logging
from typing import List, Optional

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
    model: str = "voyage-01"
    batch_size: int = 7
    show_progress_bar: bool = False

    def __init__(
        self,
        model: str,
        voyage_api_key: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
    ):
        self.model = model

        if embed_batch_size is None:
            embed_batch_size = 72 if self.model in ["voyage-2", "voyage-02"] else 7

        self.batch_size = embed_batch_size
        self.client = Client(api_key=voyage_api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        embeddings: List[List[float]] = []

        if self.show_progress_bar:
            try:
                from tqdm.auto import tqdm
            except ImportError as e:
                raise ImportError(
                    "Must have tqdm installed if `show_progress_bar` is set to True. "
                    "Please install with `pip install tqdm`."
                ) from e

            _iter = tqdm(range(0, len(texts), self.batch_size))
        else:
            _iter = range(0, len(texts), self.batch_size)

        for _i in _iter:
            embeddings_iter = self.client.embed(
                texts, model=self.model, input_type="document"
            ).embeddings
            embeddings.extend(embeddings_iter)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.client.embed(
            [text], model=self.model, input_type="query"
        ).embeddings[0]
