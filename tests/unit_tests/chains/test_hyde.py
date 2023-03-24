"""Test HyDE."""
from typing import List

import numpy as np

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.embeddings.base import Embeddings
from tests.unit_tests.llms.fake_llm import FakeLLM


class FakeEmbeddings(Embeddings):
    """Fake embedding class for tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return random floats."""
        return [list(np.random.uniform(0, 1, 10)) for _ in range(10)]

    def embed_query(self, text: str) -> List[float]:
        """Return random floats."""
        return list(np.random.uniform(0, 1, 10))


def test_hyde_from_llm() -> None:
    """Test loading HyDE from all prompts."""
    for key in PROMPT_MAP:
        embedding = HypotheticalDocumentEmbedder.from_llm(
            FakeLLM(), FakeEmbeddings(), key
        )
        embedding.embed_query("foo")


def test_hyde_from_llm_with_multiple_n() -> None:
    """Test loading HyDE from all prompts."""
    for key in PROMPT_MAP:
        embedding = HypotheticalDocumentEmbedder.from_llm(
            FakeLLM(n=8), FakeEmbeddings(), key
        )
        embedding.embed_query("foo")
