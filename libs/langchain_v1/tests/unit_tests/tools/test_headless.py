"""Tests for headless (interrupting) tools."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import BaseModel, Field

from langchain.tools import HEADLESS_TOOL_METADATA_KEY, HeadlessTool, tool


class _MessageArgs(BaseModel):
    message: str = Field(..., description="A message.")


def test_headless_tool_properties() -> None:
    t = tool(
        name="test_tool",
        description="A test headless tool.",
        args_schema=_MessageArgs,
    )
    assert isinstance(t, HeadlessTool)
    assert t.name == "test_tool"
    assert t.description == "A test headless tool."
    assert t.metadata == {HEADLESS_TOOL_METADATA_KEY: True}


def test_tool_headless_overload() -> None:
    t = tool(
        name="from_overload",
        description="via unified tool()",
        args_schema=_MessageArgs,
    )
    assert isinstance(t, HeadlessTool)
    assert t.name == "from_overload"


def test_tool_normal_still_returns_structured_tool() -> None:
    def get_weather(city: str) -> str:
        """Return a fake forecast for the city."""
        return f"sunny in {city}"

    w = tool(get_weather)
    assert not isinstance(w, HeadlessTool)
    assert w.name == "get_weather"


@pytest.mark.asyncio
async def test_headless_coroutine_calls_interrupt() -> None:
    ht = tool(
        name="interrupt_me",
        description="d",
        args_schema=_MessageArgs,
    )
    with patch("langchain.tools.headless.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "resumed"

        result = await ht.ainvoke(
            {
                "type": "tool_call",
                "name": "interrupt_me",
                "id": "call-1",
                "args": {"message": "hi"},
            }
        )

    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert payload["type"] == "tool"
    assert payload["tool_call"]["id"] == "call-1"
    assert payload["tool_call"]["name"] == "interrupt_me"
    assert payload["tool_call"]["args"] == {"message": "hi"}
    assert getattr(result, "content", result) == "resumed"


def test_headless_sync_invoke_calls_interrupt() -> None:
    """Sync `invoke` must work (StructuredTool previously had no sync path)."""
    ht = tool(
        name="sync_interrupt",
        description="d",
        args_schema=_MessageArgs,
    )
    with patch("langchain.tools.headless.interrupt") as mock_interrupt:
        mock_interrupt.return_value = "ok"
        result = ht.invoke(
            {
                "type": "tool_call",
                "name": "sync_interrupt",
                "id": "cid-9",
                "args": {"message": "sync"},
            }
        )
    mock_interrupt.assert_called_once()
    payload = mock_interrupt.call_args[0][0]
    assert payload["tool_call"]["id"] == "cid-9"
    assert getattr(result, "content", result) == "ok"


def test_headless_dict_schema_has_metadata() -> None:
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"q": {"type": "string"}},
        "required": ["q"],
    }
    ht = tool(
        name="dict_tool",
        description="Uses JSON schema.",
        args_schema=schema,
    )
    assert ht.metadata == {HEADLESS_TOOL_METADATA_KEY: True}
    assert "q" in ht.args


def test_invoke_without_graph_context_errors() -> None:
    ht = tool(
        name="t",
        description="d",
        args_schema=_MessageArgs,
    )
    with pytest.raises((RuntimeError, KeyError)):
        asyncio.run(
            ht.ainvoke(
                {
                    "type": "tool_call",
                    "name": "t",
                    "id": "x",
                    "args": {"message": "m"},
                }
            )
        )
