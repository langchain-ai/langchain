from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator

from langchain.utils import get_from_dict_or_env

logger = logging.getLogger(__name__)


class QianfanEmbeddingsEndpoint(BaseModel, Embeddings):
    """`Baidu Qianfan Embeddings` embedding models."""

    qianfan_ak: Optional[str] = None
    """Qianfan application apikey"""

    qianfan_sk: Optional[str] = None
    """Qianfan application secretkey"""

    chunk_size: int = 16
    """Chunk size when multiple texts are input"""

    model: str = "Embedding-V1"
    """Model name
    you could get from https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu
    
    for now, we support Embedding-V1 and 
    - Embedding-V1 （默认模型）
    - bge-large-en
    - bge-large-zh
    
    preset models are mapping to an endpoint.
    `model` will be ignored if `endpoint` is set
    """

    endpoint: str = ""
    """Endpoint of the Qianfan Embedding, required if custom model used."""

    client: Any
    """Qianfan client"""

    max_retries: int = 5
    """Max reties times"""

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate whether qianfan_ak and qianfan_sk in the environment variables or
        configuration file are available or not.

        init qianfan embedding client with `ak`, `sk`, `model`, `endpoint`

        Args:

            values: a dictionary containing configuration information, must include the
            fields of qianfan_ak and qianfan_sk
        Returns:

            a dictionary containing configuration information. If qianfan_ak and
            qianfan_sk are not provided in the environment variables or configuration
            file,the original values will be returned; otherwise, values containing
            qianfan_ak and qianfan_sk will be returned.
        Raises:

            ValueError: qianfan package not found, please install it with `pip install
            qianfan`
        """
        values["qianfan_ak"] = get_from_dict_or_env(
            values,
            "qianfan_ak",
            "QIANFAN_AK",
        )
        values["qianfan_sk"] = get_from_dict_or_env(
            values,
            "qianfan_sk",
            "QIANFAN_SK",
        )

        try:
            import qianfan

            params = {
                "ak": values["qianfan_ak"],
                "sk": values["qianfan_sk"],
                "model": values["model"],
            }
            if values["endpoint"] is not None and values["endpoint"] != "":
                params["endpoint"] = values["endpoint"]
            values["client"] = qianfan.Embedding(**params)
        except ImportError:
            raise ImportError(
                "qianfan package not found, please install it with "
                "`pip install qianfan`"
            )
        return values

    def embed_query(self, text: str) -> List[float]:
        resp = self.embed_documents([text])
        return resp[0]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of text documents using the AutoVOT algorithm.

        Args:
            texts (List[str]): A list of text documents to embed.

        Returns:
            List[List[float]]: A list of embeddings for each document in the input list.
                            Each embedding is represented as a list of float values.
        """
        text_in_chunks = [
            texts[i : i + self.chunk_size]
            for i in range(0, len(texts), self.chunk_size)
        ]
        lst = []
        for chunk in text_in_chunks:
            resp = self.client.do(texts=chunk)
            lst.extend([res["embedding"] for res in resp["data"]])
        return lst

    async def aembed_query(self, text: str) -> List[float]:
        embeddings = await self.aembed_documents([text])
        return embeddings[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        text_in_chunks = [
            texts[i : i + self.chunk_size]
            for i in range(0, len(texts), self.chunk_size)
        ]
        lst = []
        for chunk in text_in_chunks:
            resp = await self.client.ado(texts=chunk)
            for res in resp["data"]:
                lst.extend([res["embedding"]])
        return lst
