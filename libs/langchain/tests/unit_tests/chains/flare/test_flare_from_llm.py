from __future__ import annotations

from typing import Iterable

import pytest
from langchain.chains.flare.base import FlareChain
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever


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

    # Walk to ensure identical instance appears (not overwritten)
    seen: set[int] = set()

    def contains(target: object, obj: object) -> bool:
        if id(obj) in seen:
            return False
        seen.add(id(obj))
        if obj is target:
            return True
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                value = getattr(obj, name)
            except AttributeError:  # attribute missing only
                continue
            if isinstance(value, (str, int, float, bool, type(None))):
                continue
            if contains(target, value):
                return True
        return False

    assert contains(supplied, chain), "Supplied ChatOpenAI not found (was overwritten)."
