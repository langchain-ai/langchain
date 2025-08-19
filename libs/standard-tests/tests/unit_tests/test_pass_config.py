# --- keep this small sys.path tweak so running inside a folder named "langchain" doesn't shadow the package ---
import os, sys
_here = os.path.abspath(os.path.dirname(__file__))
_repo_root = os.path.abspath(os.path.join(_here, ".."))
# If the repo root itself is named "langchain", remove it from sys.path so `import langchain` resolves to the installed package.
if os.path.basename(_repo_root) == "langchain" and _repo_root in sys.path:
    sys.path.remove(_repo_root)

import asyncio
import logging
from typing import Any, Dict

import pytest
from langchain.agents import BaseMultiActionAgent
from langchain.agents import AgentExecutor
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnableConfig
from langchain_core.runnables.utils import accepts_config

# Prefer the new core tool decorator; fall back if your version is older.
try:
    from langchain_core.tools import tool
except Exception:  # pragma: no cover
    from langchain.tools import tool  # type: ignore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")

SEEN: Dict[str, Any] = {}  # global capture for assertions / debug


# ---- Decorator-defined tool that accepts RunnableConfig ----
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
        f"[tool] got a={a}, metadata={SEEN['metadata']}, "
        f"tags={SEEN['tags']}, run_name={SEEN['run_name']}"
    )
    return f"ok : {a}"


# ---- Minimal agent that calls the tool once, then finishes ----
class MinimalAgent(BaseMultiActionAgent):
    @property
    def input_keys(self):
        return ["input"]

    # async plan used by _acall/ainvoke
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

    # sync plan for compatibility (Executor may call it in some versions)
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

#@pytest.mark.asyncio
async def test_minimal_agent_capture_tool_config():
    
    agent = MinimalAgent()
    executor = AgentExecutor(agent=agent, tools=[capture], verbose=False)

    cfg: RunnableConfig = {
        "metadata": {"tenant": "T1", "session_id": "S1"},
        "tags": ["unit", "tool:capture"],
        "run_name": "agent-run",
    }

    # Try your patched private path first; fall back to public ainvoke if needed
    try:
        result = await executor._acall(
            {"input": "call capture tool with input a = 5"}, config=cfg  # type: ignore[attr-defined]
        )
    except (TypeError, AttributeError):
        result = await executor.ainvoke(
            {"input": "call capture tool with input a = 5"}, config=cfg
        )

    # ----- Assertions -----
    assert isinstance(result, dict)
    assert result.get("output") == "done"  # agent finishes after the tool call
    assert SEEN.get("a") == 7              # MinimalAgent hardcodes a=7

    # These validate that your executor correctly threads RunnableConfig to the tool.
    assert SEEN.get("metadata") == {"tenant": "T1", "session_id": "S1"}
    assert "unit" in (SEEN.get("tags") or [])
    assert SEEN.get("run_name") == "agent-run"

if __name__ == "__main__":
    asyncio.run(test_minimal_agent_capture_tool_config())
