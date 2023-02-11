"""Test HyDE."""
from typing import List, Optional

import numpy as np
from pydantic import BaseModel

from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain.llms.fake import FakeDictLLM
from langchain.schema import Generation, LLMResult


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
            FakeDictLLM(), FakeEmbeddings(), key
        )
        embedding.embed_query("foo")


def test_hyde_from_llm_with_multiple_n() -> None:
    """Test loading HyDE from all prompts."""
    for key in PROMPT_MAP:
        embedding = HypotheticalDocumentEmbedder.from_llm(
            FakeDictLLM(n=8), FakeEmbeddings(), key
        )
        embedding.embed_query("foo")
