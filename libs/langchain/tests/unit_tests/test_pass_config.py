"""Verify that RunnableConfig is threaded to a decorator-defined tool."""

import logging
from typing import Any

import pytest
from langchain.agents import AgentExecutor, BaseMultiActionAgent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnableConfig

try:
    from langchain_core.tools import tool
except Exception:  # pragma: no cover
    from langchain.tools import tool


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

SEEN: dict[str, Any] = {}


@tool("capture")
async def capture(a: int, *, config: RunnableConfig) -> str:
    """Capture tool that records the config it receives."""
    SEEN.clear()
    SEEN.update(
        {
            "a": a,
            "metadata": (config.get("metadata") or {}),
            "tags": config.get("tags") or [],
            "run_name": config.get("run_name"),
        }
    )
    logger.info(
        "[tool] a=%s metadata=%s tags=%s run_name=%s",
        a,
        SEEN["metadata"],
        SEEN["tags"],
        SEEN["run_name"],
    )
    return f"ok : {a}"


class MinimalAgent(BaseMultiActionAgent):
    """Agent that calls the 'capture' tool once, then finishes."""

    @property
    def input_keys(self) -> list[str]:
        return ["input"]

    async def aplan(self, intermediate_steps, **kwargs):
        if not intermediate_steps:
            return [
                AgentAction(
                    tool="capture",
                    tool_input={"a": 7},
                    log="call capture",
                )
            ]
        return AgentFinish(return_values={"output": "done"}, log="finished")

    def plan(self, intermediate_steps, **kwargs):
        if not intermediate_steps:
            return [
                AgentAction(
                    tool="capture",
                    tool_input={"a": 7},
                    log="call capture",
                )
            ]
        return AgentFinish(return_values={"output": "done"}, log="finished")


@pytest.mark.asyncio
async def test_minimal_agent_capture_tool_config():
    agent = MinimalAgent()
    executor = AgentExecutor(agent=agent, tools=[capture], verbose=False)

    cfg: RunnableConfig = {
        "metadata": {"tenant": "T1", "session_id": "S1"},
        "tags": ["unit", "tool:capture"],
        "run_name": "agent-run",
    }

    try:
        result = await executor._acall(  # type: ignore[attr-defined]
            {"input": "call capture tool with input a = 5"},
            config=cfg,
        )
    except (TypeError, AttributeError):
        result = await executor.ainvoke(
            {"input": "call capture tool with input a = 5"},
            config=cfg,
        )

    assert isinstance(result, dict)
    assert result.get("output") == "done"
    assert SEEN.get("a") == 7
    assert SEEN.get("metadata") == {"tenant": "T1", "session_id": "S1"}
    assert "unit" in (SEEN.get("tags") or [])
    assert SEEN.get("run_name") == "agent-run"
