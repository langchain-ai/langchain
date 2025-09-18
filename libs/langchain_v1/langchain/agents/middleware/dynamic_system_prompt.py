"""Dynamic System Prompt Middleware.

Allows setting the system prompt dynamically right before each model invocation.
Useful when the prompt depends on the current agent state or per-invocation context.
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from langgraph.typing import ContextT

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
)

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class DynamicSystemPromptWithoutRuntime(Protocol):
    """Dynamic system prompt without runtime in call signature."""

    def __call__(self, state: AgentState) -> str:
        """Return the system prompt for the next model call."""
        ...


class DynamicSystemPromptWithRuntime(Protocol[ContextT]):
    """Dynamic system prompt with runtime in call signature."""

    def __call__(self, state: AgentState, runtime: Runtime[ContextT]) -> str:
        """Return the system prompt for the next model call."""
        ...


DynamicSystemPrompt: TypeAlias = (
    DynamicSystemPromptWithoutRuntime | DynamicSystemPromptWithRuntime[ContextT]
)


class DynamicSystemPromptMiddleware(AgentMiddleware):
    """Dynamic System Prompt Middleware.

    Allows setting the system prompt dynamically right before each model invocation.
    Useful when the prompt depends on the current agent state or per-invocation context.

    Example:
        ```python
        from langchain.agents.middleware import DynamicSystemPromptMiddleware


        class Context(TypedDict):
            user_name: str


        def system_prompt(state: AgentState, runtime: Runtime[Context]) -> str:
            user_name = runtime.context.get("user_name", "n/a")
            return (
                f"You are a helpful assistant. Always address the user by their name: {user_name}"
            )


        middleware = DynamicSystemPromptMiddleware(system_prompt)
        ```
    """

    _accepts_runtime: bool

    def __init__(
        self,
        dynamic_system_prompt: DynamicSystemPrompt[ContextT],
    ) -> None:
        """Initialize the dynamic system prompt middleware.

        Args:
            dynamic_system_prompt: Function that receives the current agent state
                and optionally runtime with context, and returns the system prompt for
                the next model call. Returns a string.
        """
        super().__init__()
        self.dynamic_system_prompt = dynamic_system_prompt
        self._accepts_runtime = "runtime" in signature(dynamic_system_prompt).parameters

    def modify_model_request(
        self,
        request: ModelRequest,
        state: AgentState,
        runtime: Runtime[ContextT],
    ) -> ModelRequest:
        """Modify the model request to include the dynamic system prompt."""
        if self._accepts_runtime:
            system_prompt = cast(
                "DynamicSystemPromptWithRuntime[ContextT]", self.dynamic_system_prompt
            )(state, runtime)
        else:
            system_prompt = cast("DynamicSystemPromptWithoutRuntime", self.dynamic_system_prompt)(
                state
            )

        request.system_prompt = system_prompt
        return request
