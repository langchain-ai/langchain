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
    from collections.abc import Awaitable, Callable, Sequence


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
            middleware = ConditionalModelSettingsMiddleware(
                conditions={lambda req: len(req.messages) > 10: {"parallel_tool_calls": False}}
            )

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


            middleware = ConditionalModelSettingsMiddleware(
                conditions={needs_sequential_execution: {"parallel_tool_calls": False}}
            )

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2],
                middleware=[middleware],
            )
            ```

        !!! example "Multiple conditions with cumulative application"

            ```python
            middleware = ConditionalModelSettingsMiddleware(
                conditions={
                    # Base setting: all long conversations
                    lambda req: len(req.messages) > 10: {"parallel_tool_calls": False},
                    # Additional setting: emergency mode (applied on top if both match)
                    lambda req: req.state.get("emergency"): {"strict": True},
                }
            )

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


            middleware = ConditionalModelSettingsMiddleware(
                conditions={lambda req: True: compute_settings}
            )

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2],
                middleware=[middleware],
            )
            ```

        !!! example "Using list of tuples for ordered conditions"

            ```python
            # Use list of tuples when order matters or when using unhashable lambdas
            middleware = ConditionalModelSettingsMiddleware(
                conditions=[
                    (lambda req: len(req.messages) > 10, {"parallel_tool_calls": False}),
                    (lambda req: req.state.get("emergency"), {"strict": True}),
                ]
            )

            agent = create_agent(
                model="openai:gpt-4o",
                tools=[tool1, tool2],
                middleware=[middleware],
            )
            ```
    """

    def __init__(
        self,
        conditions: (
            dict[
                Callable[[ModelRequest], bool | Awaitable[bool]],
                dict[str, Any] | Callable[[ModelRequest], dict[str, Any]],
            ]
            | Sequence[
                tuple[
                    Callable[[ModelRequest], bool | Awaitable[bool]],
                    dict[str, Any] | Callable[[ModelRequest], dict[str, Any]],
                ]
            ]
            | None
        ) = None,
    ) -> None:
        """Initialize middleware with conditions and settings.

        Args:
            conditions: Either a dict mapping condition functions to settings dicts,
                or a sequence of (condition, settings) tuples. Settings are merged
                with existing model_settings. If None, no conditions are registered.
        """
        super().__init__()
        if conditions is None:
            self._conditions: list[
                tuple[
                    Callable[[ModelRequest], bool | Awaitable[bool]],
                    dict[str, Any] | Callable[[ModelRequest], dict[str, Any]],
                ]
            ] = []
        elif isinstance(conditions, dict):
            self._conditions = list(conditions.items())
        else:
            self._conditions = list(conditions)

    def _merge_settings(
        self,
        request: ModelRequest,
        settings: dict[str, Any] | Callable[[ModelRequest], dict[str, Any]],
    ) -> dict[str, Any]:
        """Merge settings with existing model_settings and return merged dict."""
        resolved_settings = settings(request) if callable(settings) else settings
        return {**request.model_settings, **resolved_settings}

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Apply conditional settings before calling model."""
        merged_settings = request.model_settings

        for condition, settings in self._conditions:
            if iscoroutinefunction(condition):
                msg = (
                    "Async condition function detected in sync execution path. "
                    "Use sync condition or invoke agent with `astream()`/`ainvoke()`."
                )
                raise RuntimeError(msg)

            if condition(request):
                resolved_settings = settings(request) if callable(settings) else settings
                merged_settings = {**merged_settings, **resolved_settings}

        if merged_settings != request.model_settings:
            request = request.override(model_settings=merged_settings)

        return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Apply conditional settings before calling model (async)."""
        merged_settings = request.model_settings

        for condition, settings in self._conditions:
            if iscoroutinefunction(condition):
                result = await condition(request)
            else:
                result = condition(request)

            if result:
                resolved_settings = settings(request) if callable(settings) else settings
                merged_settings = {**merged_settings, **resolved_settings}

        if merged_settings != request.model_settings:
            request = request.override(model_settings=merged_settings)

        return await handler(request)
