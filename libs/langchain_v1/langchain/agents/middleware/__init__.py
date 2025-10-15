"""Middleware plugins for agents."""

from .context_editing import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
)
from .human_in_the_loop import (
    HumanInTheLoopMiddleware,
    InterruptOnConfig,
)
from .model_call_limit import ModelCallLimitMiddleware
from .model_fallback import ModelFallbackMiddleware
from .pii import PIIDetectionError, PIIMiddleware
from .summarization import SummarizationMiddleware
from .todo import TodoListMiddleware
from .tool_call_limit import ToolCallLimitMiddleware
from .tool_emulator import LLMToolEmulator
from .tool_selection import LLMToolSelectorMiddleware
from .types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    hook_config,
    wrap_model_call,
    wrap_tool_call,
)

__all__ = [
    "AgentMiddleware",
    "AgentState",
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
    "HumanInTheLoopMiddleware",
    "InterruptOnConfig",
    "LLMToolEmulator",
    "LLMToolSelectorMiddleware",
    "ModelCallLimitMiddleware",
    "ModelFallbackMiddleware",
    "ModelRequest",
    "ModelResponse",
    "PIIDetectionError",
    "PIIMiddleware",
    "SummarizationMiddleware",
    "TodoListMiddleware",
    "ToolCallLimitMiddleware",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "wrap_model_call",
    "wrap_tool_call",
]
