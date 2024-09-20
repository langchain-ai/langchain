from difflib import SequenceMatcher
from typing import List, Tuple

from pydantic import BaseModel

from langchain_community.cross_encoders.base import BaseCrossEncoder


class FakeCrossEncoder(BaseCrossEncoder, BaseModel):
    """Fake cross encoder model."""

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        scores = list(
            map(
                lambda pair: SequenceMatcher(None, pair[0], pair[1]).ratio(), text_pairs
            )
        )
        return scores
