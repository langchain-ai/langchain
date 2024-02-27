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

    def __init__(
        self,
        model: str = "voyage-01",
        voyage_api_key: Optional[str] = None,
        embed_batch_size: Optional[int] = None,
    ):
        if model == "voyage-01":
            logger.warning(
                "voyage-01 is not the latest model by Voyage AI. Please note that "
                "`model_name` will be a required argument in the future. We recommend "
                "setting it explicitly. Please see "
                "https://docs.voyageai.com/docs/embeddings for the latest models "
                "offered by Voyage AI."
            )
        self.model = model

        if embed_batch_size is None:
            embed_batch_size = 72 if self.model in ["voyage-2", "voyage-02"] else 7

        self.batch_size = embed_batch_size
        self.client = Client(api_key=voyage_api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        return self.client.embed(
            texts, model=self.model, input_type="document"
        ).embeddings

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.client.embed(
            [text], model=self.model, input_type="query"
        ).embeddings[0]
