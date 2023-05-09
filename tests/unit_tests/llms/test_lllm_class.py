"""Fake LLM wrapper for testing purposes."""
import asyncio
from typing import Any, List, Mapping, Optional, cast

import pytest

from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.schema import LLMResult


class FakeNewSignLLM(LLM):
    """Fake LLM wrapper for testing purposes."""

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
    ) -> str:
        return prompt

    async def _acall(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    ) -> str:
        return prompt

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "fake_test_only"

def test_llm_intern_generate():
    llm = FakeNewSignLLM()
    results = llm._generate(["hello", "world"])
    assert isinstance(results, LLMResult)
    assert len(results.generations) == 2
    assert results.generations[0][0].text == "hello"
    assert results.generations[1][0].text == "world"

@pytest.mark.asyncio
async def test_llm_async_intern_generate():
    llm = FakeNewSignLLM()
    results = await llm._agenerate(["hello", "world"])
    assert isinstance(results, LLMResult)
    assert len(results.generations) == 2
    assert results.generations[0][0].text == "hello"
    assert results.generations[1][0].text == "world"