from typing import Any, Dict

from langchain.embeddings.base import Embeddings
from langchain.schema import BaseOutputParser


class SQLCommandOutputParser(BaseOutputParser[Dict[str, Any]]):
    @property
    def _type(self) -> str:
        return "sql_bypass"

    def parse(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        return {"llm_out": text, "sql_cmd": text}


class VectorSQLOutputParser(SQLCommandOutputParser):
    """Output Parser for Vector SQL
    1. finds for `NeuralArray()` and replace it with the embedding
    2. finds for `DISTANCE()` and replace it with the distance name in backend SQL
    """

    model: Embeddings
    """Embedding model to extract embedding for entity"""
    distance_func_name: str = "distance"
    """Distance name for vector SQL"""

    class Config:
        arbitrary_types_allowed = 1

    @property
    def _type(self) -> str:
        return "vector_sql"

    @classmethod
    def from_embeddings(
        cls, model: Embeddings, distance_func_name: str = "distance", **kwargs: Any
    ) -> BaseOutputParser:
        return cls(model=model, distance_func_name=distance_func_name, **kwargs)

    def parse(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        start = text.find("NeuralArray(")
        _sql_str_compl = text
        if start > 0:
            _matched = text[text.find("NeuralArray(") + len("NeuralArray(") :]
            end = _matched.find(")") + start + len("NeuralArray(") + 1
            entity = _matched[: _matched.find(")")]
            vecs = self.model.embed_query(entity)
            vecs_str = "[" + ",".join(map(str, vecs)) + "]"
            _sql_str_compl = text.replace("DISTANCE", self.distance_func_name).replace(
                text[start:end], vecs_str
            )
            if _sql_str_compl[-1] == ";":
                _sql_str_compl = _sql_str_compl[:-1]
        return {"llm_out": text, "sql_cmd": _sql_str_compl}


class VectorSQLRetrieveAllOutputParser(VectorSQLOutputParser):
    """Based on VectorSQLOutputParser
    It also modify the SQL to get all columns
    """

    @property
    def _type(self) -> str:
        return "vector_sql_retrieve_all"

    def parse(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        start = text.upper().find("SELECT")
        if start >= 0:
            end = text.upper().find("FROM")
            text = text.replace(text[start + len("SELECT") + 1 : end - 1], "*")
        return super().parse(text)
