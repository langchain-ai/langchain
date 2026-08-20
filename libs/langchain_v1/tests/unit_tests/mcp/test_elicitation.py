"""Unit tests for the MCP elicitation -> interrupt bridge."""

from __future__ import annotations

from typing import Any

import pytest
from langchain_core.tools import ToolException
from mcp.types import (
    CallToolResult,
    ElicitRequest,
    ElicitRequestFormParams,
    InputRequiredResult,
    TextContent,
)

from langchain.mcp import _elicitation

_SCHEMA = {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}


class _Session:
    """Stub MCP session: asks for input once, then returns a terminal result."""

    def __init__(self) -> None:
        self.rounds = 0

    async def call_tool(
        self,
        name: str,  # noqa: ARG002
        arguments: dict[str, Any],  # noqa: ARG002
        *,
        allow_input_required: bool = False,  # noqa: ARG002
        input_responses: dict[str, Any] | None = None,
        request_state: str | None = None,  # noqa: ARG002
    ) -> InputRequiredResult | CallToolResult:
        if input_responses is None:
            self.rounds += 1
            return InputRequiredResult(
                result_type="input_required",
                request_state="s1",
                input_requests={
                    "date": ElicitRequest(
                        method="elicitation/create",
                        params=ElicitRequestFormParams(
                            mode="form", message="What date?", requested_schema=_SCHEMA
                        ),
                    )
                },
            )
        assert input_responses["date"].action == "accept"
        return CallToolResult(content=[TextContent(type="text", text="booked")], is_error=False)


class _Client:
    def __init__(self) -> None:
        self.session = _Session()


async def test_elicitation_bridges_to_interrupt(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[Any] = []

    def fake_interrupt(payload: Any) -> dict[str, Any]:
        seen.append(payload)
        return {"action": "accept", "content": {"date": "2026-12-24"}}

    monkeypatch.setattr(_elicitation, "interrupt", fake_interrupt)

    client: Any = _Client()
    result = await _elicitation.call_tool_with_elicitation(client, "book", {})

    block = result.content[0]
    assert isinstance(block, TextContent)
    assert block.text == "booked"
    # The interrupt payload surfaces the elicitation for a UI / resume handler.
    assert seen[0]["type"] == "mcp_elicitation"
    assert seen[0]["message"] == "What date?"
    assert seen[0]["response_schema"] == _SCHEMA


async def test_invalid_resume_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_elicitation, "interrupt", lambda _payload: {"action": "maybe"})
    client: Any = _Client()
    with pytest.raises(ToolException, match="Invalid elicitation response"):
        await _elicitation.call_tool_with_elicitation(client, "book", {})
