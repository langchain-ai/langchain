from dataclasses import dataclass
from typing import List


@dataclass
class CohereCitation:
    """
    TODO
    """

    start: int
    end: int
    text: str
    documents: List


def format_text_with_citations(text: str, citations: List[CohereCitation]) -> str:
    return ""
