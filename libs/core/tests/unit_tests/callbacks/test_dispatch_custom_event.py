import sys
import uuid
from typing import Any
from uuid import UUID

import pytest

from langchain_core.callbacks import AsyncCallbackHandler, BaseCallbackHandler
from langchain_core.callbacks.manager import (
    adispatch_custom_event,
    dispatch_custom_event,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.config import RunnableConfig


class AsyncCustomCallbackHandler(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.events: list[Any] = []

    async def on_custom_event(
        self,
        name: str,
        data: Any,
        *,
        run_id: UUID,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        assert kwargs == {}
        self.events.append(
            (
                name,
                data,
                run_id,
                tags,
                metadata,
            )
        )


def test_custom_event_root_dispatch() -> None:
    """Test adhoc event in a nested chain."""
    # This just tests that nothing breaks on the path.
    # It shouldn't do anything at the moment, since the tracer isn't configured
    # to handle adhoc events.
    # Expected behavior is that the event cannot be dispatched
    with pytest.raises(RuntimeError):
        dispatch_custom_event("event1", {"x": 1})


async def test_async_custom_event_root_dispatch() -> None:
    """Test adhoc event in a nested chain."""
    # This just tests that nothing breaks on the path.
    # It shouldn't do anything at the moment, since the tracer isn't configured
    # to handle adhoc events.
    # Expected behavior is that the event cannot be dispatched
    with pytest.raises(RuntimeError):
        await adispatch_custom_event("event1", {"x": 1})


IS_GTE_3_11 = sys.version_info >= (3, 11)


@pytest.mark.skipif(not IS_GTE_3_11, reason="Requires Python >=3.11")
async def test_async_custom_event_implicit_config() -> None:
    """Test dispatch without passing config explicitly."""
    callback = AsyncCustomCallbackHandler()

    run_id = uuid.UUID(int=7)

    @RunnableLambda
    async def foo(x: int, config: RunnableConfig) -> int:
        assert "callbacks" in config
        await adispatch_custom_event("event1", {"x": x})
        await adispatch_custom_event("event2", {"x": x})
        return x

    await foo.ainvoke(
        1,
        {"callbacks": [callback], "run_id": run_id},
    )

    assert callback.events == [
        ("event1", {"x": 1}, UUID("00000000-0000-0000-0000-000000000007"), [], {}),
        ("event2", {"x": 1}, UUID("00000000-0000-0000-0000-000000000007"), [], {}),
    ]


async def test_async_callback_manager() -> None:
    """Test async callback manager."""
    callback = AsyncCustomCallbackHandler()

    run_id = uuid.UUID(int=7)

    @RunnableLambda
    async def foo(x: int, config: RunnableConfig) -> int:
        await adispatch_custom_event("event1", {"x": x}, config=config)
        await adispatch_custom_event("event2", {"x": x}, config=config)
        return x

    await foo.ainvoke(
        1,
        {"callbacks": [callback], "run_id": run_id},
    )

    assert callback.events == [
        ("event1", {"x": 1}, UUID("00000000-0000-0000-0000-000000000007"), [], {}),
        ("event2", {"x": 1}, UUID("00000000-0000-0000-0000-000000000007"), [], {}),
    ]


def test_sync_callback_manager() -> None:
    """Test async callback manager."""

    class CustomCallbackManager(BaseCallbackHandler):
        def __init__(self) -> None:
            self.events: list[Any] = []

        def on_custom_event(
            self,
            name: str,
            data: Any,
            *,
            run_id: UUID,
            tags: list[str] | None = None,
            metadata: dict[str, Any] | None = None,
            **kwargs: Any,
        ) -> None:
            assert kwargs == {}
            self.events.append(
                (
                    name,
                    data,
                    run_id,
                    tags,
                    metadata,
                )
            )

    callback = CustomCallbackManager()

    run_id = uuid.UUID(int=7)

    @RunnableLambda
    def foo(x: int, config: RunnableConfig) -> int:
        dispatch_custom_event("event1", {"x": x})
        dispatch_custom_event("event2", {"x": x}, config=config)
        return x

    foo.invoke(1, {"callbacks": [callback], "run_id": run_id})

    assert callback.events == [
        ("event1", {"x": 1}, UUID("00000000-0000-0000-0000-000000000007"), [], {}),
        ("event2", {"x": 1}, UUID("00000000-0000-0000-0000-000000000007"), [], {}),
    ]
