from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel, root_validator
from typing import Any, Dict, List, Optional
from volcengine.viking_db import *

from langchain_core.utils import get_from_dict_or_env


class VikingDBEmbeddings(BaseModel, Embeddings):

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["ak"] = get_from_dict_or_env(
            values,
            "ak",
            "AK",
        )
        values["sk"] = get_from_dict_or_env(
            values,
            "sk",
            "SK",
        )
        values["host"] = get_from_dict_or_env(
            values,
            "host",
            "HOST",
        )
        values["region"] = get_from_dict_or_env(
            values,
            "region",
            "REGION",
        )
        values["viking_db"] = VikingDBService(values["host"], values["region"], values["ak"], values["sk"])
        values["emb_model"] = EmbModel("bge_large_zh")
        return values

    def embed_query(self, text: str) -> List[float]:

        raw_data = RawData("text", text)
        return self.viking_db.embedding(self.emb_model, raw_data)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        raw_datas = []
        for item in texts:
            raw_data = RawData("text", item)
            raw_datas.append(raw_data)
        return self.viking_db.embedding(self.emb_model, raw_datas)
