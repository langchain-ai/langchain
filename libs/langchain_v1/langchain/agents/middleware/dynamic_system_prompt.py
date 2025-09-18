"""Dynamic System Prompt Middleware.

Allows setting the system prompt dynamically right before each model invocation.
Useful when the prompt depends on the current agent state or per-invocation context.
"""

from __future__ import annotations

from inspect import signature
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from langchain_core.messages import SystemMessage
from langgraph.typing import ContextT

from langchain.agents.middleware.types import AgentMiddleware, AgentState, ModelRequest

if TYPE_CHECKING:
    from langgraph.runtime import Runtime


class DynamicSystemPromptWithoutRuntime(Protocol):
    """Dynamic system prompt without runtime in call signature."""

    def __call__(self, state: AgentState) -> str | SystemMessage:
        """Return the system prompt for the next model call."""
        ...


class DynamicSystemPromptWithRuntime(Protocol[ContextT]):
    """Dynamic system prompt with runtime in call signature."""

    def __call__(self, state: AgentState, runtime: Runtime[ContextT]) -> str | SystemMessage:
        """Return the system prompt for the next model call."""
        ...


DynamicSystemPrompt: TypeAlias = DynamicSystemPromptWithoutRuntime | DynamicSystemPromptWithRuntime


class DynamicSystemPromptMiddleware(AgentMiddleware[AgentState, ContextT]):
    """Dynamic System Prompt Middleware.

    Allows setting the system prompt dynamically right before each model invocation.
    Useful when the prompt depends on the current agent state or per-invocation context.

    Example:
        ```python
        from langchain.agents.middleware.dynamic_system_prompt import DynamicSystemPromptMiddleware
        from langchain_core.messages import SystemMessage


        def get_system_prompt(state: AgentState, runtime: Runtime) -> str:
            region = runtime.context.get("region", "n/a")
            return f"You are a helpful assistant. Region: {region}"


        middleware = DynamicSystemPromptMiddleware(get_system_prompt)
        ```

    Example with SystemMessage:
        ```python
        def get_system_message(state: AgentState, runtime: Runtime) -> SystemMessage:
            region = runtime.context.get("region", "n/a")
            return SystemMessage(content=f"You are a helpful assistant. Region: {region}")


        middleware = DynamicSystemPromptMiddleware(get_system_message)
        ```
    """

    def __init__(
        self,
        dynamic_system_prompt: DynamicSystemPrompt,
    ) -> None:
        """Initialize the dynamic system prompt middleware.

        Args:
            dynamic_system_prompt: Function that receives the current agent state and runtime,
                and returns the system prompt for the next model call. It can return either a
                SystemMessage or a string (which will be wrapped in a SystemMessage).
        """
        super().__init__()
        self.dynamic_system_prompt = dynamic_system_prompt
        sig = signature(dynamic_system_prompt)
        self.accepts_runtime = "runtime" in sig.parameters

    def modify_model_request(
        self,
        request: ModelRequest,
        state: AgentState,
        runtime: Runtime[ContextT],
    ) -> ModelRequest:
        """Modify the model request to include the dynamic system prompt."""
        if self.accepts_runtime:
            system_prompt = cast(
                "DynamicSystemPromptWithRuntime[ContextT]", self.dynamic_system_prompt
            )(state, runtime)
        else:
            system_prompt = cast("DynamicSystemPromptWithoutRuntime", self.dynamic_system_prompt)(
                state
            )

        request.system_prompt = (
            str(system_prompt.content)
            if isinstance(system_prompt, SystemMessage)
            else system_prompt
        )
        return request
