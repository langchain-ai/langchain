"""Middleware for automatically adding provider-specific builtin tools.

This middleware enables easy switching between LLM providers while using native
server-side tools like web_search, code_execution, etc. It automatically detects
the provider and injects the appropriate tool format.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, Literal

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelCallResult, ModelResponse

logger = logging.getLogger(__name__)

# Tool registry mapping canonical tool names to provider-specific formats
BUILTIN_TOOL_REGISTRY: dict[str, dict[str, dict[str, Any] | None]] = {
    # Web Search
    "web_search": {
        "anthropic": {"type": "web_search_20250305", "name": "web_search"},
        "openai": {"type": "web_search"},
        "google_genai": {"google_search": {}},
        "xai": None,  # Configured via search_parameters on model
    },
    # Code Execution
    "code_execution": {
        "anthropic": {"type": "code_execution_20250825", "name": "code_execution"},
        "openai": {"type": "code_interpreter", "container": {"type": "auto"}},
        "google_genai": {"code_execution": {}},
        "xai": None,
    },
    # Web Fetch (Anthropic only)
    "web_fetch": {
        "anthropic": {"type": "web_fetch_20250910", "name": "web_fetch"},
        "openai": None,
        "google_genai": None,
        "xai": None,
    },
    # Memory (Anthropic only)
    "memory": {
        "anthropic": {"type": "memory_20250818", "name": "memory"},
        "openai": None,
        "google_genai": None,
        "xai": None,
    },
    # File Search (OpenAI only - requires vector_store_ids)
    "file_search": {
        "anthropic": None,
        "openai": {"type": "file_search"},  # User must add vector_store_ids
        "google_genai": None,
        "xai": None,
    },
    # Image Generation (OpenAI only)
    "image_generation": {
        "anthropic": None,
        "openai": {"type": "image_generation"},
        "google_genai": None,
        "xai": None,
    },
    # Text Editor (Anthropic only)
    "text_editor": {
        "anthropic": {"type": "text_editor_20250728", "name": "str_replace_based_edit_tool"},
        "openai": None,
        "google_genai": None,
        "xai": None,
    },
    # Bash (Anthropic only)
    "bash": {
        "anthropic": {"type": "bash_20250124", "name": "bash"},
        "openai": None,
        "google_genai": None,
        "xai": None,
    },
}


def _detect_provider(model: Any) -> str:
    """Detect the provider from the model instance.

    Args:
        model: The chat model instance.

    Returns:
        Provider identifier string.
    """
    model_class_name = model.__class__.__name__

    # Check for specific model classes
    if "Anthropic" in model_class_name:
        return "anthropic"
    if "OpenAI" in model_class_name or "AzureChatOpenAI" == model_class_name:
        return "openai"
    if "Google" in model_class_name or "Vertex" in model_class_name:
        return "google_genai"
    if "XAI" in model_class_name or model_class_name == "ChatXAI":
        return "xai"

    # Fallback: return unknown
    return "unknown"


class BuiltinToolsMiddleware(AgentMiddleware):
    """Middleware that automatically adds provider-specific builtin tools.

    This middleware intercepts model calls and injects builtin tool definitions
    based on the detected model provider. It abstracts away the different tool
    formats required by Anthropic, OpenAI, and Google.

    Examples:
        !!! example "Basic usage with common tools"

            ```python
            from langchain.agents.middleware import BuiltinToolsMiddleware
            from langchain.agents import create_agent

            middleware = BuiltinToolsMiddleware(
                include_tools=["web_search", "code_execution"]
            )

            agent = create_agent(
                model="openai:gpt-4o",
                middleware=[middleware],
            )
            ```

        !!! example "With tool-specific options"

            ```python
            middleware = BuiltinToolsMiddleware(
                include_tools=[
                    "web_search",
                    {"name": "web_fetch", "max_uses": 5},
                    {"name": "file_search", "vector_store_ids": ["vs_123"]},
                ],
                unsupported_behavior="warn",
            )
            ```

        !!! example "Error on unsupported tools"

            ```python
            middleware = BuiltinToolsMiddleware(
                include_tools=["web_fetch"],  # Anthropic only
                unsupported_behavior="error",  # Will error if used with OpenAI
            )
            ```
    """

    def __init__(
        self,
        include_tools: list[str | dict[str, Any]],
        *,
        unsupported_behavior: Literal["remove", "error", "warn"] = "remove",
    ) -> None:
        """Initialize the builtin tools middleware.

        Args:
            include_tools: List of tool names or tool specifications to include.

                Can be:

                - Simple tool name: `"web_search"`
                - Tool with options: `{"name": "web_fetch", "max_uses": 5}`
            unsupported_behavior: How to handle tools not supported by a provider.

                - `"remove"` (default): Silently skip unsupported tools
                - `"error"`: Raise `ValueError` during model call if any tool is
                    unsupported
                - `"warn"`: Log warning and skip unsupported tools
        """
        super().__init__()
        self.include_tools = include_tools
        self.unsupported_behavior = unsupported_behavior

    def _parse_tool_spec(self, tool_spec: str | dict[str, Any]) -> tuple[str, dict[str, Any]]:
        """Parse a tool specification into name and options.

        Args:
            tool_spec: Tool name string or dict with name and options.

        Returns:
            Tuple of (tool_name, options_dict).
        """
        if isinstance(tool_spec, str):
            return tool_spec, {}

        tool_name = tool_spec.get("name")
        if not tool_name:
            msg = f"Tool specification must have 'name' key: {tool_spec}"
            raise ValueError(msg)

        # Extract all keys except 'name' as options
        options = {k: v for k, v in tool_spec.items() if k != "name"}
        return tool_name, options

    def _build_tool_definition(
        self,
        tool_name: str,
        provider: str,
        options: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Build provider-specific tool definition.

        Args:
            tool_name: Canonical tool name.
            provider: Provider identifier.
            options: Additional options to merge into the tool definition.

        Returns:
            Provider-specific tool definition dict, or None if unsupported.
        """
        if tool_name not in BUILTIN_TOOL_REGISTRY:
            logger.warning(f"Unknown builtin tool: {tool_name}")
            return None

        provider_tools = BUILTIN_TOOL_REGISTRY[tool_name]
        base_tool = provider_tools.get(provider)

        if base_tool is None:
            return None

        # Deep copy and merge options
        tool_def = base_tool.copy()

        # Merge user-provided options into the tool definition
        if options:
            # For nested structures, we need to merge carefully
            for key, value in options.items():
                if key in tool_def and isinstance(tool_def[key], dict) and isinstance(value, dict):
                    # Merge nested dicts
                    tool_def[key] = {**tool_def[key], **value}
                else:
                    # Direct assignment
                    tool_def[key] = value

        return tool_def

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        """Inject builtin tools before calling the model.

        Args:
            request: Model request to modify.
            handler: Handler to execute the model request.

        Returns:
            Model response from the handler.

        Raises:
            ValueError: If unsupported_behavior is "error" and a tool is unsupported.
        """
        # Detect provider
        provider = _detect_provider(request.model)

        if provider == "unknown":
            logger.warning(
                f"Could not detect provider for model {request.model.__class__.__name__}. "
                "Builtin tools will not be added."
            )
            return handler(request)

        # Build tool definitions
        builtin_tools: list[dict[str, Any]] = []
        unsupported_tools: list[str] = []

        for tool_spec in self.include_tools:
            tool_name, options = self._parse_tool_spec(tool_spec)
            tool_def = self._build_tool_definition(tool_name, provider, options)

            if tool_def is None:
                unsupported_tools.append(tool_name)
            else:
                builtin_tools.append(tool_def)

        # Handle unsupported tools
        if unsupported_tools:
            unsupported_str = ", ".join(unsupported_tools)
            if self.unsupported_behavior == "error":
                msg = (
                    f"The following builtin tools are not supported by {provider}: "
                    f"{unsupported_str}"
                )
                raise ValueError(msg)
            if self.unsupported_behavior == "warn":
                warnings.warn(
                    f"The following builtin tools are not supported by {provider} "
                    f"and will be skipped: {unsupported_str}",
                    stacklevel=2,
                )

        # If no tools to add, just call handler
        if not builtin_tools:
            return handler(request)

        # Merge builtin tools with existing tools
        # Existing tools should come first, then builtin tools
        combined_tools = list(request.tools) + builtin_tools

        # Create modified request with builtin tools
        modified_request = request.override(tools=combined_tools)

        return handler(modified_request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        """Async version of wrap_model_call.

        Args:
            request: Model request to modify.
            handler: Async handler to execute the model request.

        Returns:
            Model response from the handler.

        Raises:
            ValueError: If unsupported_behavior is "error" and a tool is unsupported.
        """
        # Detect provider
        provider = _detect_provider(request.model)

        if provider == "unknown":
            logger.warning(
                f"Could not detect provider for model {request.model.__class__.__name__}. "
                "Builtin tools will not be added."
            )
            return await handler(request)

        # Build tool definitions
        builtin_tools: list[dict[str, Any]] = []
        unsupported_tools: list[str] = []

        for tool_spec in self.include_tools:
            tool_name, options = self._parse_tool_spec(tool_spec)
            tool_def = self._build_tool_definition(tool_name, provider, options)

            if tool_def is None:
                unsupported_tools.append(tool_name)
            else:
                builtin_tools.append(tool_def)

        # Handle unsupported tools
        if unsupported_tools:
            unsupported_str = ", ".join(unsupported_tools)
            if self.unsupported_behavior == "error":
                msg = (
                    f"The following builtin tools are not supported by {provider}: "
                    f"{unsupported_str}"
                )
                raise ValueError(msg)
            if self.unsupported_behavior == "warn":
                warnings.warn(
                    f"The following builtin tools are not supported by {provider} "
                    f"and will be skipped: {unsupported_str}",
                    stacklevel=2,
                )

        # If no tools to add, just call handler
        if not builtin_tools:
            return await handler(request)

        # Merge builtin tools with existing tools
        combined_tools = list(request.tools) + builtin_tools

        # Create modified request with builtin tools
        modified_request = request.override(tools=combined_tools)

        return await handler(modified_request)

