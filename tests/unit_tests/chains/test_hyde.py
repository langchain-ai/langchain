"""Test HyDE."""
from typing import Any, List, Optional

import numpy as np

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain.chains.hyde.prompts import PROMPT_MAP
from langchain.embeddings.base import Embeddings
from langchain.llms.base import BaseLLM
from langchain.schema import Generation, LLMResult


class FakeEmbeddings(Embeddings):
    """Fake embedding class for tests."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return random floats."""
        return [list(np.random.uniform(0, 1, 10)) for _ in range(10)]

    def embed_query(self, text: str) -> List[float]:
        """Return random floats."""
        return list(np.random.uniform(0, 1, 10))


class FakeLLM(BaseLLM):
    """Fake LLM wrapper for testing purposes."""

    n: int = 1

    def _generate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text="foo") for _ in range(self.n)]])

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text="foo") for _ in range(self.n)]])

    def get_num_tokens(self, text: str) -> int:
        """Return number of tokens."""
        return len(text.split())

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake"


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
