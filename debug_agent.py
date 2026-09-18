"""Debug entry point for the `create_agent` -> model -> tool -> model loop.

默认使用离线的 `FakeToolCallingModel`，不需要任何 API Key，也不需要联网。

推荐断点（按执行顺序）：

1. libs/langchain_v1/langchain/agents/factory.py
   - `create_agent`            : agent 构建，看 tools / middleware
   - `_get_bound_model`        : 决定给模型绑定哪些工具
   - `_execute_model_sync`     : 进入模型调用
   - `model_to_tools`          : 判断要不要跳到 tools 节点（返回 Send）
   - `tools_to_model`          : 工具执行完，判断回 model 还是结束
2. libs/core/langchain_core/language_models/chat_models.py
   - `BaseChatModel.invoke` / `_generate_with_cache`
3. libs/langchain_v1/.venv/Lib/site-packages/langgraph/prebuilt/tool_node.py
   - `ToolNode._func` / `_run_one` / `_execute_tool_sync` / `_inject_tool_args`
   （langgraph 是外部依赖，断点必然落在 venv 里，这是正常的）
4. libs/core/langchain_core/tools/base.py
   - `BaseTool.invoke` / `BaseTool.run` / `_parse_input` / `_format_output`
5. libs/core/langchain_core/tools/structured.py
   - `StructuredTool._run`     : 下一行就进入你自己的工具函数

切换成真实模型（可选）：
    设置环境变量 DEBUG_AGENT_MODEL="openai:gpt-4o-mini" 和 OPENAI_API_KEY
    （需要先安装 langchain-openai，例如在 libs/langchain_v1 下执行 uv sync --group test）
"""

from __future__ import annotations

import os
from typing import Any, Literal

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from typing_extensions import override

from langchain.agents import create_agent

USER_INPUT = "北京现在天气怎么样？"
SYSTEM_PROMPT = "You are a helpful assistant."


# --------------------------------------------------------------------------- #
# 1. 离线假模型：第 1 轮返回带 tool_calls 的 AIMessage，第 2 轮返回最终答案
#    逻辑与 tests/unit_tests/agents/model.py 的 FakeToolCallingModel 保持一致
# --------------------------------------------------------------------------- #
class FakeToolCallingModel(BaseChatModel):
    """A deterministic chat model that replays a fixed list of tool calls."""

    tool_calls: list[list[dict[str, Any]]] = []
    index: int = 0
    tool_style: Literal["openai", "anthropic"] = "openai"

    @property
    def _llm_type(self) -> str:
        return "fake-tool-call-model"

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        # 在这里观察 `messages`：第 1 轮只有 HumanMessage，
        # 第 2 轮会多出 AIMessage(tool_calls) 和 ToolMessage。
        calls = self.tool_calls[self.index] if self.index < len(self.tool_calls) else []
        message = AIMessage(
            content="-".join(m.text for m in messages),
            id=str(self.index),
            tool_calls=[dict(c) for c in calls],
        )
        self.index += 1
        return ChatResult(generations=[ChatGeneration(message=message)])

    @override
    def bind_tools(
        self,
        tools: list[dict[str, Any] | type | Any] | tuple[Any, ...],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Any:
        tool_dicts: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                tool_dicts.append(t)
                continue
            if not isinstance(t, BaseTool):
                msg = "Only BaseTool and dict are supported by FakeToolCallingModel.bind_tools"
                raise TypeError(msg)
            if self.tool_style == "openai":
                tool_dicts.append({"type": "function", "function": {"name": t.name}})
            else:
                tool_dicts.append({"name": t.name})
        return self.bind(tools=tool_dicts, **kwargs)


# --------------------------------------------------------------------------- #
# 2. 工具：`@tool` 会把普通函数转成 StructuredTool
#    在 `StructuredTool._run` 里 `self.func(*args, **kwargs)` 就是这里
# --------------------------------------------------------------------------- #
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city.

    Returns:
        A short weather description.
    """
    return f"{city}: 25C, sunny"


@tool
def get_local_time(city: str) -> str:
    """Get the local time for a city.

    Args:
        city: The name of the city.

    Returns:
        The local time as a string.
    """
    return f"{city}: 14:30"


def build_agent() -> Any:
    """Build the agent, either with a real model or the offline fake one."""
    tools = [get_weather, get_local_time]
    model_spec = os.environ.get("DEBUG_AGENT_MODEL")
    if model_spec:
        return create_agent(model_spec, tools=tools, system_prompt=SYSTEM_PROMPT)
    return create_agent(
        FakeToolCallingModel(
            tool_calls=[
                [{"name": "get_weather", "args": {"city": "Beijing"}, "id": "call_1"}],
                [],
            ]
        ),
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )


def print_messages(messages: list[BaseMessage]) -> None:
    """Print a compact trace of the final message list."""
    print("\n===== final messages =====")
    for i, m in enumerate(messages):
        tool_calls = getattr(m, "tool_calls", None)
        tool_call_id = getattr(m, "tool_call_id", None)
        print(f"[{i}] {type(m).__name__}")
        if tool_calls:
            print(f"      tool_calls={tool_calls}")
        if tool_call_id:
            print(f"      tool_call_id={tool_call_id}")
        content = m.text if m.text else m.content
        print(f"      content={content!r}")


def stream_trace(agent: Any) -> None:
    """Print graph-level debug events (no breakpoints needed)."""
    print("\n===== stream_mode='debug' =====")
    for event in agent.stream({"messages": [HumanMessage(USER_INPUT)]}, stream_mode="debug"):
        payload = event.get("payload") or {}
        name = payload.get("name") or payload.get("id") or ""
        print(f"{event.get('type'):<14} {event.get('step'):<3} {name}")


def main() -> None:
    agent = build_agent()

    if os.environ.get("DEBUG_AGENT_STREAM"):
        stream_trace(agent)
        return

    result = agent.invoke({"messages": [HumanMessage(USER_INPUT)]})
    print_messages(result["messages"])


if __name__ == "__main__":
    main()
