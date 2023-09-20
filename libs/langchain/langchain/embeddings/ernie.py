import logging
import warnings
from typing import Any, Dict, List, Optional

from langchain.embeddings.base import Embeddings
from langchain.pydantic_v1 import BaseModel, root_validator
from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class ErnieEmbeddings(BaseModel, Embeddings):
    """`Ernie Embeddings V1` embedding models.

    This embedding is the implement only about Baidu ERNIE EmbeddingV1.
    Use Baidu `langchain.embeddings.QianfanEmbeddingsEndpoint` for more
    embedding models.
    """

    ernie_client_id: Optional[str] = None
    ernie_client_secret: Optional[str] = None
    access_token: Optional[str] = None

    chunk_size: int = 16

    model_name = "ErnieBot-Embedding-V1"

    client: Any

    def __new__(cls, **data: Any) -> "ErnieEmbeddings":
        """Initialize the ErnieEmbeddings object."""
        warnings.warn(
            "You are trying to use a ERNIE embedding model. This way of initializing it"
            "is a subset of QianfanEmbeddingsEndpoint which not only provides ERNIE "
            "Embedding-V1 more embeddings but also more embedding models. Instead use: "
            "`from langchain.embeddings import QianfanEmbeddingsEndpoint`"
        )
        return super().__new__(cls)

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["ernie_client_id"] = get_from_dict_or_env(
            values,
            "ernie_client_id",
            "ERNIE_CLIENT_ID",
        )
        values["ernie_client_secret"] = get_from_dict_or_env(
            values,
            "ernie_client_secret",
            "ERNIE_CLIENT_SECRET",
        )
        try:
            import erniebot

            erniebot.ak = values["ernie_client_id"]
            erniebot.sk = values["ernie_client_secret"]

            values["client"] = erniebot.Embedding
        except ImportError:
            raise ValueError(
                "erniebot package not found, please install it with "
                "`pip install erniebot`"
            )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        text_in_chunks = [
            texts[i : i + self.chunk_size]
            for i in range(0, len(texts), self.chunk_size)
        ]
        lst = []
        for chunk in text_in_chunks:
            resp = self.client.create(
                **{"model": "ernie-text-embedding", "input": chunk}
            )
            lst.extend([i["embedding"] for i in resp["data"]])
        return lst

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents(texts=[text])[0]
