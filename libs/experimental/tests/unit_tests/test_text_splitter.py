import re
from typing import List

import pytest
from langchain_core.embeddings import Embeddings

from langchain_experimental.text_splitter import SemanticChunker

FAKE_EMBEDDINGS = [
    [0.02905, 0.42969, 0.65394, 0.62200],
    [0.00515, 0.47214, 0.45327, 0.75605],
    [0.57401, 0.30344, 0.41702, 0.63603],
    [0.60308, 0.18708, 0.68871, 0.35634],
    [0.52510, 0.56163, 0.34100, 0.54089],
    [0.73275, 0.22089, 0.42652, 0.48204],
    [0.47466, 0.26161, 0.79687, 0.26694],
]
SAMPLE_TEXT = (
    "We need to harvest synergy effects viral engagement, but digitalize, "
    "nor overcome key issues to meet key milestones. So digital literacy "
    "where the metal hits the meat. So this vendor is incompetent. Can "
    "you champion this? Let me diarize this. And we can synchronise "
    "ourselves at a later timepoint t-shaped individual tread it daily. "
    "That is a good problem"
)


class MockEmbeddings(Embeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return FAKE_EMBEDDINGS[: len(texts)]

    def embed_query(self, text: str) -> List[float]:
        return [1.0, 2.0]


@pytest.mark.parametrize(
    "input_length, expected_length",
    [
        (1, 1),
        (2, 2),
        (5, 2),
    ],
)
def test_split_text_gradient(input_length: int, expected_length: int) -> None:
    embeddings = MockEmbeddings()
    chunker = SemanticChunker(
        embeddings,
        breakpoint_threshold_type="gradient",
    )
    list_of_sentences = re.split(r"(?<=[.?!])\s+", SAMPLE_TEXT)[:input_length]

    chunks = chunker.split_text(" ".join(list_of_sentences))

    assert len(chunks) == expected_length
