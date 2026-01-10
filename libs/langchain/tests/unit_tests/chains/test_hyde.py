"""Test HyDE."""

from typing import Any

import numpy as np
from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import BaseLLM
from langchain_core.outputs import Generation, LLMResult
from typing_extensions import override

from langchain_classic.chains.hyde.base import HypotheticalDocumentEmbedder
from langchain_classic.chains.hyde.prompts import PROMPT_MAP


class FakeEmbeddings(Embeddings):
    """Fake embedding class for tests."""

    @override
    def embed_documents(self, texts: list[str], **_kwargs: Any) -> list[list[float]]:
        """Return random floats."""
        return [list(np.random.default_rng().uniform(0, 1, 10)) for _ in range(10)]

    @override
    def embed_query(self, text: str, **_kwargs: Any) -> list[float]:
        """Return random floats."""
        return list(np.random.default_rng().uniform(0, 1, 10))


class FakeLLM(BaseLLM):
    """Fake LLM wrapper for testing purposes."""

    n: int = 1

    @override
    def _generate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> LLMResult:
        return LLMResult(generations=[[Generation(text="foo") for _ in range(self.n)]])

    @override
    async def _agenerate(
        self,
        prompts: list[str],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
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
            FakeLLM(),
            FakeEmbeddings(),
            key,
        )
        embedding.embed_query("foo")


def test_hyde_from_llm_with_multiple_n() -> None:
    """Test loading HyDE from all prompts."""
    for key in PROMPT_MAP:
        embedding = HypotheticalDocumentEmbedder.from_llm(
            FakeLLM(n=8),
            FakeEmbeddings(),
            key,
        )
        embedding.embed_query("foo")
