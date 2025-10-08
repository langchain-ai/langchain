"""Middleware plugins for agents."""

from .context_editing import (
    ClearToolUsesEdit,
    ContextEditingMiddleware,
)
from .human_in_the_loop import HumanInTheLoopMiddleware
from .model_call_limit import ModelCallLimitMiddleware
from .model_fallback import ModelFallbackMiddleware
from .pii import PIIDetectionError, PIIMiddleware
from .planning import PlanningMiddleware
from .prompt_caching import AnthropicPromptCachingMiddleware
from .summarization import SummarizationMiddleware
from .tool_call_limit import ToolCallLimitMiddleware
from .tool_selection import LLMToolSelectorMiddleware
from .types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    after_agent,
    after_model,
    before_agent,
    before_model,
    dynamic_prompt,
    hook_config,
    modify_model_request,
    on_model_call,
)

__all__ = [
    "AgentMiddleware",
    "AgentState",
    # should move to langchain-anthropic if we decide to keep it
    "AnthropicPromptCachingMiddleware",
    "ClearToolUsesEdit",
    "ContextEditingMiddleware",
    "HumanInTheLoopMiddleware",
    "LLMToolSelectorMiddleware",
    "ModelCallLimitMiddleware",
    "ModelFallbackMiddleware",
    "ModelRequest",
    "PIIDetectionError",
    "PIIMiddleware",
    "PlanningMiddleware",
    "SummarizationMiddleware",
    "ToolCallLimitMiddleware",
    "after_agent",
    "after_model",
    "before_agent",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "modify_model_request",
    "on_model_call",
]
