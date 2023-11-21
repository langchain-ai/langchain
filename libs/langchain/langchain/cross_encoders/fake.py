from difflib import SequenceMatcher
from typing import List

from langchain.pydantic_v1 import BaseModel
from langchain.schema.cross_encoder import CrossEncoder


class FakeCrossEncoder(CrossEncoder, BaseModel):
    """Fake cross encoder model."""

    def score(self, pairs: List[List[str]]) -> List[float]:
        scores = list(
            map(lambda pair: SequenceMatcher(None, pair[0], pair[1]).ratio(), pairs)
        )
        return scores
