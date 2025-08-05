import asyncio
from itertools import cycle
from typing import Any, Optional, Union
from uuid import UUID

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]
from typing_extensions import override

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.language_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from langchain_core.v1.messages import AIMessageChunk as AIMessageChunkV1
from langchain_core.v1.messages import MessageV1


class MyCustomAsyncHandler(AsyncCallbackHandler):
    @override
    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: Union[list[list[BaseMessage]], list[MessageV1]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        # Do nothing
        # Required to implement since this is an abstract method
        pass

    @override
    async def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: Optional[
            Union[GenerationChunk, ChatGenerationChunk, AIMessageChunkV1]
        ] = None,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        await asyncio.sleep(0)


@pytest.mark.benchmark
async def test_async_callbacks_in_sync(benchmark: BenchmarkFixture) -> None:
    infinite_cycle = cycle([AIMessage(content=" ".join(["hello", "goodbye"] * 5))])
    model = GenericFakeChatModel(messages=infinite_cycle)

    @benchmark  # type: ignore[misc]
    def sync_callbacks() -> None:
        for _ in range(5):
            for _ in model.stream("meow", {"callbacks": [MyCustomAsyncHandler()]}):
                pass
