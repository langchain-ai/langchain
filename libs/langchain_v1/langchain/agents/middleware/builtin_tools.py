"""Middleware for automatically adding provider-specific builtin tools.

This middleware enables easy switching between LLM providers while using native
server-side tools like web_search, code_execution, etc. It automatically detects
the provider and injects the appropriate tool format.

The middleware uses standard TypedDict schemas from langchain_core.tools.builtin
and converts them to provider-specific formats using conversion utilities from
each provider's integration package.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from importlib import import_module, util
from typing import TYPE_CHECKING, Any, Literal, cast

from langchain.agents.middleware.types import AgentMiddleware, ModelRequest

if TYPE_CHECKING:
    from collections.abc import Awaitable

    from langchain.agents.middleware.types import ModelCallResult, ModelResponse

logger = logging.getLogger(__name__)

# Converter callable type
Converter = Callable[[dict[str, Any]], dict[str, Any] | None]

# Literal type for all supported builtin tool names
BuiltinToolName = Literal[
    "web_search",
    "code_execution",
    "x_search",
    "web_fetch",
    "memory",
    "file_search",
    "image_generation",
    "text_editor",
    "bash",
]


def _load_converter(
    *,
    package: str,
    converter_module: str,
    converter_name: str,
    tool_name: str,
    package_display: str,
) -> Converter | None:
    """Load a converter function from an optional dependency."""
    if not util.find_spec(package):
        logger.debug(
            "%s package not installed, cannot convert tool: %s",
            package_display,
            tool_name,
        )
        return None

    try:
        module = import_module(converter_module)
    except ImportError:
        logger.debug(
            "%s package not installed, cannot convert tool: %s",
            package_display,
            tool_name,
        )
        return None

    converter = getattr(module, converter_name, None)
    if converter is None:
        logger.debug(
            "%s.%s not found, cannot convert tool: %s",
            converter_module,
            converter_name,
            tool_name,
        )
        return None

    return cast("Converter", converter)


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
    if "OpenAI" in model_class_name or model_class_name == "AzureChatOpenAI":
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
    formats required by Anthropic, OpenAI, Google, and xAI (Grok).

    Examples:
        !!! example "Basic usage with common tools"

            ```python
            from langchain.agents.middleware import BuiltinToolsMiddleware
            from langchain.agents import create_agent

            middleware = BuiltinToolsMiddleware(include_tools=["web_search", "code_execution"])

            agent = create_agent(
                model="openai:gpt-4o",
                middleware=[middleware],
            )
            ```

        !!! example "Using xAI/Grok with X search"

            ```python
            middleware = BuiltinToolsMiddleware(
                include_tools=["web_search", "x_search", "code_execution"]
            )

            agent = create_agent(
                model="xai:grok-2-latest",
                middleware=[middleware],
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
        *,
        include_tools: list[BuiltinToolName],
        unsupported_behavior: Literal["remove", "error", "warn"] = "remove",
    ) -> None:
        """Initialize the builtin tools middleware.

        Args:
            include_tools: List of tool names to include.

                Supported tool names:
                - `"web_search"`: Web search (OpenAI, Anthropic, xAI)
                - `"code_execution"`: Code interpreter (OpenAI, Anthropic, Google, xAI)
                - `"x_search"`: X/Twitter search (xAI only)
                - `"web_fetch"`: Web page fetching (Anthropic only)
                - `"memory"`: Memory/context management (Anthropic only)
                - `"file_search"`: File search (OpenAI only)
                - `"image_generation"`: Image generation (OpenAI only)
                - `"text_editor"`: Text editor (Anthropic only)
                - `"bash"`: Bash execution (Anthropic only)
            unsupported_behavior: How to handle tools not supported by a provider.

                - `"remove"` (default): Silently skip unsupported tools
                - `"error"`: Raise `ValueError` during model call if any tool is
                    unsupported
                - `"warn"`: Log warning and skip unsupported tools
        """
        super().__init__()
        self.include_tools = include_tools
        self.unsupported_behavior = unsupported_behavior

    def _build_tool_definition(
        self,
        tool_name: BuiltinToolName,
        provider: str,
    ) -> dict[str, Any] | None:
        """Build provider-specific tool definition.

        Creates a standard builtin tool definition and converts it to the
        provider-specific format using conversion utilities from the provider's
        integration package.

        Args:
            tool_name: Canonical tool name (e.g., "web_search", "code_execution").
            provider: Provider identifier (e.g., "openai", "anthropic", "xai").

        Returns:
            Provider-specific tool definition dict, or None if unsupported by provider
            or if provider package is not installed.
        """
        # Create standard tool definition
        standard_tool: dict[str, Any] = {"type": tool_name}

        # Convert to provider-specific format using conversion utilities
        if provider == "openai":
            converter = _load_converter(
                package="langchain_openai",
                converter_module="langchain_openai.utils.builtin_tools",
                converter_name="convert_standard_to_openai",
                tool_name=tool_name,
                package_display="langchain-openai",
            )
            if converter is None:
                return None

            return converter(standard_tool)

        if provider == "anthropic":
            converter = _load_converter(
                package="langchain_anthropic",
                converter_module="langchain_anthropic.utils.builtin_tools",
                converter_name="convert_standard_to_anthropic",
                tool_name=tool_name,
                package_display="langchain-anthropic",
            )
            if converter is None:
                return None

            return converter(standard_tool)

        if provider == "xai":
            converter = _load_converter(
                package="langchain_xai",
                converter_module="langchain_xai.utils.builtin_tools",
                converter_name="convert_standard_to_xai",
                tool_name=tool_name,
                package_display="langchain-xai",
            )
            if converter is None:
                return None

            return converter(standard_tool)

        if provider == "google_genai":
            # Google GenAI uses a different format that doesn't fit the standard pattern
            # For now, handle it directly here
            if tool_name == "web_search":
                return {"google_search": {}}
            if tool_name == "code_execution":
                return {"code_execution": {}}
            # Other tools not supported by Google GenAI
            return None

        # Unknown provider
        logger.warning("Unknown provider: %s", provider)
        return None

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
                "Could not detect provider for model %s. Builtin tools will not be added.",
                request.model.__class__.__name__,
            )
            return handler(request)

        # Build tool definitions
        builtin_tools: list[dict[str, Any]] = []
        unsupported_tools: list[str] = []

        for tool_name in self.include_tools:
            tool_def = self._build_tool_definition(tool_name, provider)

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
                "Could not detect provider for model %s. Builtin tools will not be added.",
                request.model.__class__.__name__,
            )
            return await handler(request)

        # Build tool definitions
        builtin_tools: list[dict[str, Any]] = []
        unsupported_tools: list[str] = []

        for tool_name in self.include_tools:
            tool_def = self._build_tool_definition(tool_name, provider)

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
