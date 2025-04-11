# ruff: noqa: ARG002
import asyncio
from itertools import cycle
from typing import Any

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage


class MyCustomAsyncHandler(AsyncCallbackHandler):
    async def on_chat_model_start(
        self,
        serialized: Any,
        messages: Any,
        *,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        metadata: Any = None,
        **kwargs: Any,
    ) -> Any:
        # Do nothing
        # Required to implement since this is an abstract method
        pass

    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Any = None,
        run_id: Any,
        parent_run_id: Any = None,
        tags: Any = None,
        **kwargs: Any,
    ) -> None:
        await asyncio.sleep(0)


@pytest.mark.benchmark
async def test_async_callbacks(benchmark: BenchmarkFixture) -> None:
    infinite_cycle = cycle([AIMessage(content=" ".join(["hello", "goodbye"] * 1000))])
    model = GenericFakeChatModel(messages=infinite_cycle)

    @benchmark
    def async_callbacks() -> None:
        for _ in range(10):
            for _ in model.stream("meow", {"callbacks": [MyCustomAsyncHandler()]}):
                pass
