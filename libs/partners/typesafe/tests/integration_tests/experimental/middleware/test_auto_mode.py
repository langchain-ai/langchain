"""Live integration tests for `AutoModeMiddleware`."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain.agents.middleware.types import InputAgentState
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.tools import tool
from typing_extensions import Self, override

from langchain_typesafe.experimental.middleware import AutoModeMiddleware


class _ToolCallingModel(GenericFakeChatModel):
    """Deterministic chat model that accepts tool binding."""

    @override
    def bind_tools(
        self,
        tools: Sequence[Any],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Self:
        """Return this model after accepting the agent's tools."""
        _ = (tools, tool_choice, kwargs)
        return self


@pytest.mark.parametrize("async_", [False, True])
async def test_live_classification_blocks_agent_tool_execution(
    *,
    async_: bool,
) -> None:
    """Block a tool through complete synchronous and asynchronous agent runs."""
    executions: list[str] = []

    @tool
    def delete_file(path: str) -> str:
        """Delete a file at the supplied path."""
        executions.append(path)
        return "deleted"

    model = _ToolCallingModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        ToolCall(
                            name="delete_file",
                            args={"path": "/workspace/report.txt"},
                            id="call_live",
                            type="tool_call",
                        )
                    ],
                ),
                AIMessage("done"),
            ]
        )
    )
    middleware = AutoModeMiddleware(tools=[delete_file])
    agent = create_agent(model, tools=[delete_file], middleware=[middleware])
    state = InputAgentState(messages=[HumanMessage("Summarize the report.")])

    try:
        if async_:
            result = await agent.ainvoke(state)
        else:
            result = agent.invoke(state)

        tool_messages = [
            message
            for message in result["messages"]
            if isinstance(message, ToolMessage)
        ]
        [tool_message] = tool_messages
        assert tool_message.status == "error"
        assert "was blocked because it was classified as risky" in tool_message.text
        assert tool_message.tool_call_id == "call_live"
        assert executions == []
    finally:
        if middleware.classifier.async_client is not None:
            await middleware.classifier.async_client.aclose()
        if middleware.classifier.client is not None:
            middleware.classifier.client.close()
