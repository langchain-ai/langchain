from collections.abc import AsyncGenerator, Generator
from typing import Any, cast

import pytest
from langchain_core.runnables import RunnableSerializable

from langchain_classic.agents.agent import RunnableAgent


class _DictStreamRunnable(RunnableSerializable[dict, dict]):
    """Streams dict chunks to emulate structured agent outputs."""

    def stream(  # type: ignore[override]
        self, _: dict, config: dict | None = None
    ) -> Generator[dict, None, None]:
        assert config is not None  # ensure config is accepted
        yield {"a": "foo"}
        yield {"a": "bar"}
        yield {"b": 1}

    async def astream(  # type: ignore[override]
        self, _: dict, config: dict | None = None
    ) -> AsyncGenerator[dict, None]:
        assert config is not None
        yield {"a": "foo"}
        yield {"a": "bar"}
        yield {"b": 1}

    def invoke(self, _: dict, config: dict | None = None) -> dict:  # type: ignore[override]
        assert config is not None
        return {"invoke": True}

    async def ainvoke(  # type: ignore[override]
        self, _: dict, config: dict | None = None
    ) -> dict:
        assert config is not None
        return {"ainvoke": True}


def test_runnable_agent_stream_combines_dict_chunks() -> None:
    agent = RunnableAgent(
        runnable=_DictStreamRunnable(),
        input_keys_arg=["messages"],
        return_keys_arg=[],
    )

    result = cast("dict[str, Any]", agent.plan(intermediate_steps=[], messages=[]))

    assert result == {"a": "foobar", "b": 1}


@pytest.mark.asyncio
async def test_runnable_agent_astream_combines_dict_chunks() -> None:
    agent = RunnableAgent(
        runnable=_DictStreamRunnable(),
        input_keys_arg=["messages"],
        return_keys_arg=[],
    )

    result = cast(
        "dict[str, Any]", await agent.aplan(intermediate_steps=[], messages=[])
    )

    assert result == {"a": "foobar", "b": 1}
