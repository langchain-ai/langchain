"""Middleware plugins for agents."""

from .human_in_the_loop import HumanInTheLoopMiddleware
from .model_fallback import ModelFallbackMiddleware
from .pii import PIIDetectionError, PIIMiddleware
from .planning import PlanningMiddleware
from .prompt_caching import AnthropicPromptCachingMiddleware
from .summarization import SummarizationMiddleware
from .tool_call_limit import ToolCallLimitMiddleware
from .types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    after_model,
    before_model,
    dynamic_prompt,
    hook_config,
    modify_model_request,
)

__all__ = [
    "AgentMiddleware",
    "AgentState",
    # should move to langchain-anthropic if we decide to keep it
    "AnthropicPromptCachingMiddleware",
    "HumanInTheLoopMiddleware",
    "ModelFallbackMiddleware",
    "ModelRequest",
    "PIIDetectionError",
    "PIIMiddleware",
    "PlanningMiddleware",
    "SummarizationMiddleware",
    "ToolCallLimitMiddleware",
    "after_model",
    "before_model",
    "dynamic_prompt",
    "hook_config",
    "modify_model_request",
]
