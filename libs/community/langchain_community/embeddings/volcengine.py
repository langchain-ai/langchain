from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from langchain_core.embeddings import Embeddings
from langchain_core.utils import get_from_dict_or_env, pre_init
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class VolcanoEmbeddings(BaseModel, Embeddings):
    """`Volcengine Embeddings` embedding models."""

    volcano_ak: Optional[str] = None
    """volcano access key
    learn more from: https://www.volcengine.com/docs/6459/76491#ak-sk"""

    volcano_sk: Optional[str] = None
    """volcano secret key
    learn more from: https://www.volcengine.com/docs/6459/76491#ak-sk"""

    host: str = "maas-api.ml-platform-cn-beijing.volces.com"
    """host
    learn more from https://www.volcengine.com/docs/82379/1174746"""
    region: str = "cn-beijing"
    """region
    learn more from https://www.volcengine.com/docs/82379/1174746"""

    model: str = "bge-large-zh"
    """Model name
    you could get from https://www.volcengine.com/docs/82379/1174746
    for now, we support bge_large_zh
    """

    version: str = "1.0"
    """ model version """

    chunk_size: int = 100
    """Chunk size when multiple texts are input"""

    client: Any
    """volcano client"""

    @pre_init
    def validate_environment(cls, values: Dict) -> Dict:
        """
        Validate whether volcano_ak and volcano_sk in the environment variables or
        configuration file are available or not.

        init volcano embedding client with `ak`, `sk`, `host`, `region`

        Args:

            values: a dictionary containing configuration information, must include the
            fields of volcano_ak and volcano_sk
        Returns:

            a dictionary containing configuration information. If volcano_ak and
            volcano_sk are not provided in the environment variables or configuration
            file,the original values will be returned; otherwise, values containing
            volcano_ak and volcano_sk will be returned.
        Raises:

            ValueError: volcengine package not found, please install it with
            `pip install volcengine`
        """
        values["volcano_ak"] = get_from_dict_or_env(
            values,
            "volcano_ak",
            "VOLC_ACCESSKEY",
        )
        values["volcano_sk"] = get_from_dict_or_env(
            values,
            "volcano_sk",
            "VOLC_SECRETKEY",
        )

        try:
            from volcengine.maas import MaasService

            client = MaasService(values["host"], values["region"])
            client.set_ak(values["volcano_ak"])
            client.set_sk(values["volcano_sk"])
            values["client"] = client
        except ImportError:
            raise ImportError(
                "volcengine package not found, please install it with "
                "`pip install volcengine`"
            )
        return values

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

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
            req = {
                "model": {
                    "name": self.model,
                    "version": self.version,
                },
                "input": chunk,
            }
            try:
                from volcengine.maas import MaasException

                resp = self.client.embeddings(req)
                lst.extend([res["embedding"] for res in resp["data"]])
            except MaasException as e:
                raise ValueError(f"embed by volcengine Error: {e}")
        return lst
