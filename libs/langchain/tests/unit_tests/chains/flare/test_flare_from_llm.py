import pytest

from langchain.chains.flare.base import FlareChain
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from typing import List


class _EmptyRetriever(BaseRetriever):  # Minimal retriever for tests
    def _get_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        return []

    async def _aget_relevant_documents(self, query: str) -> List[Document]:  # type: ignore[override]
        return []


def test_from_llm_rejects_non_chatopenai():
    class Dummy:
        pass

    with pytest.raises(TypeError):
        FlareChain.from_llm(Dummy())  # type: ignore[arg-type]


def test_from_llm_uses_supplied_chatopenai(monkeypatch):
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
        retriever=_EmptyRetriever(),  # Provide required field
    )

    # Walk to ensure identical instance appears (not overwritten)
    seen = set()

    def contains(target, obj):
        if id(obj) in seen:
            return False
        seen.add(id(obj))
        if obj is target:
            return True
        for name in dir(obj):
            if name.startswith("__"):
                continue
            try:
                v = getattr(obj, name)
            except Exception:  # noqa: BLE001
                continue
            if isinstance(v, (str, int, float, bool, type(None))):
                continue
            if contains(target, v):
                return True
        return False

    assert contains(supplied, chain), "Supplied ChatOpenAI not found (was overwritten)."
