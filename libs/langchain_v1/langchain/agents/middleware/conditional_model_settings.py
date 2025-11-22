"""Conditional model settings middleware for dynamic configuration."""

from __future__ import annotations

from inspect import iscoroutinefunction
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware.types import (
    AgentMiddleware,
    ModelCallResult,
    ModelRequest,
    ModelResponse,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable


class _ConditionBuilder:
    """Builder for fluent API: middleware.when(condition).use(settings)."""

    def __init__(
        self,
        middleware: ConditionalModelSettingsMiddleware,
        condition: Callable[[ModelRequest], bool | Awaitable[bool]],
    ) -> None:
        self._middleware = middleware
        self._condition = condition

    def use(
        self,
        settings: dict[str, Any] | Callable[[ModelRequest], dict[str, Any]],
    ) -> ConditionalModelSettingsMiddleware:
        """Apply settings when condition is met.

        Args:
            settings: Dict of model settings or callable returning settings dict.

        Returns:
            Parent middleware instance for chaining.
        """
        self._middleware._conditions.append((self._condition, settings))
        return self._middleware


class ConditionalModelSettingsMiddleware(AgentMiddleware):
    """Dynamically configure model bind settings based on runtime conditions.

    This middleware allows you to apply different `model_settings` (passed to
    `model.bind_tools()` or `model.bind()`) based on conditions evaluated at runtime.
    All matching conditions have their settings applied cumulatively (later settings
    override earlier ones for the same keys).

    Note: `model_settings` are parameters passed to the model's bind method, such as
    `parallel_tool_calls`, `strict`, etc. For model inference parameters like
    `temperature`, `max_tokens`, use the model configuration directly in `create_agent`.

    Examples:
        !!! example "Simple usage - disable parallel tool calls"

            ```python
            from langchain.agents import create_agent
            from langchain.agents.middleware import ConditionalModelSettingsMiddleware

            # Disable parallel tool calls for long conversations
            middleware = ConditionalModelSettingsMiddleware()
            middleware.when(lambda req: len(req.messages) > 10).use({"parallel_tool_calls": False})

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[some_tool],
                middleware=[middleware],
            )
            ```

        !!! example "Using function for condition"

            ```python
            def needs_sequential_execution(req: ModelRequest) -> bool:
                # Check if state indicates sequential execution needed
                return req.state.get("execution_mode") == "sequential"


            middleware = ConditionalModelSettingsMiddleware()
            middleware.when(needs_sequential_execution).use({"parallel_tool_calls": False})

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2],
                middleware=[middleware],
            )
            ```

        !!! example "Multiple conditions with cumulative application"

            ```python
            middleware = ConditionalModelSettingsMiddleware()

            # Base setting: all long conversations
            middleware.when(lambda req: len(req.messages) > 10).use({"parallel_tool_calls": False})

            # Additional setting: emergency mode (applied on top if both match)
            middleware.when(lambda req: req.state.get("emergency")).use({"strict": True})

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2, tool3],
                middleware=[middleware],
            )

            # If messages > 10 AND emergency=True:
            # Result: {"parallel_tool_calls": False, "strict": True}
            # Both conditions apply cumulatively
            ```

        !!! example "Dynamic settings with callable"

            ```python
            def compute_settings(req: ModelRequest) -> dict[str, Any]:
                # Enable parallel calls only for short conversations
                if len(req.messages) < 5:
                    return {"parallel_tool_calls": True}
                return {"parallel_tool_calls": False}


            middleware = ConditionalModelSettingsMiddleware()
            middleware.when(lambda req: True).use(compute_settings)

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2],
                middleware=[middleware],
            )
            ```
    """

    def __init__(self) -> None:
        """Initialize middleware. Settings are merged with existing model_settings."""
        super().__init__()
        self._conditions: list[
            tuple[
                Callable[[ModelRequest], bool | Awaitable[bool]],
                dict[str, Any] | Callable[[ModelRequest], dict[str, Any]],
            ]
        ] = []

    def when(
        self,
        condition: Callable[[ModelRequest], bool | Awaitable[bool]],
    ) -> _ConditionBuilder:
        """Register condition for applying settings.

        Args:
            condition: Callable taking ModelRequest and returning bool (sync or async).

        Returns:
            Builder object with .use() method.
        """
        return _ConditionBuilder(self, condition)

    def _apply_settings(
        self,
        request: ModelRequest,
        settings: dict[str, Any] | Callable[[ModelRequest], dict[str, Any]],
    ) -> None:
        """Apply settings to request."""
        resolved_settings = settings(request) if callable(settings) else settings
        request.model_settings = {**request.model_settings, **resolved_settings}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply conditional settings before calling model."""
        for condition, settings in self._conditions:
            if iscoroutinefunction(condition):
                msg = (
                    "Async condition function detected in sync execution path. "
                    "Use sync condition or invoke agent with `astream()`/`ainvoke()`."
                )
                raise RuntimeError(msg)

            if condition(request):
                self._apply_settings(request, settings)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply conditional settings before calling model (async)."""
        for condition, settings in self._conditions:
            if iscoroutinefunction(condition):
                result = await condition(request)
            else:
                result = condition(request)

            if result:
                self._apply_settings(request, settings)

        return await handler(request)
