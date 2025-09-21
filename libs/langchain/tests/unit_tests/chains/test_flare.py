"""Tests for FlareChain.from_llm preserving supplied ChatOpenAI instance."""

from typing import cast

import pydantic
import pydantic._internal._model_construction
import pytest
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableSequence

from langchain.chains.flare.base import FlareChain

# patch validation for Pydantic 2.7 (issue due to OpenAI SDK)
try:
    import pydantic._internal._model_construction

    # Simple patch: make field validation always return False to avoid conflicts
    pydantic._internal._model_construction.is_valid_field_name = (  # type: ignore[attr-defined]
        lambda name: False  # noqa: ARG005
    )
except (AttributeError, ImportError):
    pass


class _EmptyRetriever(BaseRetriever):
    """Minimal no-op retriever used only for constructing FlareChain in tests."""

    def _get_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        del query  # mark used
        return []

    async def _aget_relevant_documents(self, query: str) -> list[Document]:  # type: ignore[override]
        del query  # mark used
        return []


def test_from_llm_rejects_non_chatopenai() -> None:
    class Dummy:
        pass

    with pytest.raises(TypeError):
        FlareChain.from_llm(Dummy())  # type: ignore[arg-type]


@pytest.mark.requires("langchain_openai")
def test_from_llm_uses_supplied_chatopenai(monkeypatch: pytest.MonkeyPatch) -> None:
    try:
        from langchain_openai import ChatOpenAI
    except ImportError:  # pragma: no cover
        pytest.skip("langchain-openai not installed")

    # Provide dummy API key to satisfy constructor env validation.
    monkeypatch.setenv("OPENAI_API_KEY", "TEST")

    supplied = ChatOpenAI(temperature=0.51, logprobs=True, max_completion_tokens=21)
    chain = FlareChain.from_llm(
        supplied,
        max_generation_len=32,
        retriever=_EmptyRetriever(),
    )

    llm_in_chain = cast("RunnableSequence", chain.question_generator_chain).steps[1]
    assert llm_in_chain is supplied
