"""Reproduce response_format incompatibility with natural text streaming.

This script demonstrates that `create_agent(..., response_format=...)` can suppress
natural assistant text streaming when `ToolStrategy` is used, because the agent
binds the model with `tool_choice="any"`.

Run from this package directory:
    uv run python scripts/repro_streaming_response_format.py

This script uses a local fake streaming chat model and does not require network.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel, Field
from typing_extensions import override

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable, Iterator, Sequence

    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langchain_core.runnables import Runnable


logger = logging.getLogger(__name__)


class WeatherReport(BaseModel):
    """Weather response."""

    temperature: float = Field(description="The temperature in fahrenheit")
    condition: str = Field(description="Weather condition")


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"The weather in {city} is sunny and 75°F."


class StreamingToolCallingModel(BaseChatModel):
    """A fake chat model that can stream either text or tool-call chunks.

    This model is intentionally simplistic:
    - On the first call, it emits a tool call to `get_weather`.
    - On the second call, it emits a final assistant response.

    When bound with `tool_choice="any"` (ToolStrategy), it suppresses text streaming
    and emits only tool call chunks.
    """

    step: int = 0
    tool_choice_seen: str | None = None

    @property
    @override
    def _llm_type(self) -> str:
        return "streaming-tool-calling-fake"

    @override
    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tool_choice_seen = tool_choice

        tool_dicts: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                tool_dicts.append(t)
                continue
            if not isinstance(t, BaseTool):
                msg = "Only BaseTool and dict are supported by StreamingToolCallingModel.bind_tools"
                raise TypeError(msg)
            tool_dicts.append({"type": "function", "function": {"name": t.name}})

        return self.bind(tools=tool_dicts, **kwargs)

    def _next_message(self) -> AIMessage:
        if self.step == 0:
            self.step += 1
            return AIMessage(
                content="I will check the weather now.",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "get_weather",
                        "args": {"city": "Boston"},
                    }
                ],
            )

        self.step += 1
        return AIMessage(content="It is sunny and 75°F in Boston.")

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        _ = run_manager
        _ = (messages, stop, kwargs)
        msg = self._next_message()
        return ChatResult(generations=[ChatGeneration(message=msg)])

    @override
    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        _ = run_manager
        _ = (messages, stop, kwargs)
        msg = self._next_message()

        if self.tool_choice_seen == "any" and msg.tool_calls:
            # Simulate "tool-call-first" streaming:
            # - no natural language text tokens
            # - but still produce a valid tool call by the last chunk so the tool executes.
            yield ChatGenerationChunk(message=AIMessageChunk(content=""))
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content="",
                    tool_calls=[
                        {
                            "id": "call_1",
                            "name": "get_weather",
                            "args": {"city": "Boston"},
                        }
                    ],
                    chunk_position="last",
                )
            )
            return

        # Natural text streaming mode.
        text = str(msg.content)
        for idx, ch in enumerate(text):
            yield ChatGenerationChunk(
                message=AIMessageChunk(
                    content=ch,
                    chunk_position=("last" if idx == len(text) - 1 else None),
                    tool_calls=msg.tool_calls if idx == len(text) - 1 else [],
                )
            )

    @override
    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        _ = run_manager
        for chunk in self._stream(messages, stop=stop, **kwargs):
            yield chunk
            await asyncio.sleep(0)


async def _run_case(*, title: str, response_format: Any | None) -> None:
    model = StreamingToolCallingModel()
    agent = create_agent(
        model,
        tools=[get_weather],
        system_prompt="You are a helpful weather assistant.",
        response_format=response_format,
    )

    logger.info("\n%s\n%s\n%s", "=" * 80, title, "=" * 80)

    text_chunks: list[str] = []
    tool_chunks: list[str] = []

    async for msg, meta in agent.astream(
        {"messages": [HumanMessage(content="What's the weather in Boston?")]},
        stream_mode="messages",
    ):
        _ = meta
        for block in getattr(msg, "content_blocks", None) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text" and block.get("text"):
                text_chunks.append(str(block["text"]))
            if block.get("type") == "tool_call_chunk" and block.get("args"):
                tool_chunks.append(str(block["args"]))

    logger.info("tool_choice_seen=%r", model.tool_choice_seen)
    logger.info("text_streamed=%r", "".join(text_chunks))
    logger.info("tool_args_streamed=%r", "".join(tool_chunks))


async def main() -> None:
    """Run local reproduction cases."""
    await _run_case(title="Baseline (no response_format)", response_format=None)
    await _run_case(
        title="ToolStrategy (response_format via ToolStrategy)",
        response_format=ToolStrategy(WeatherReport),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
