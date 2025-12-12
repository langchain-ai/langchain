"""Convenience wrappers for Anthropic beta tool TypedDicts.

These wrappers auto-populate the required literal fields (name, type) so you
don't have to specify them manually.

Example:
    Instead of:
        from anthropic.types.beta import BetaToolBash20250124Param
        tool = BetaToolBash20250124Param(
            name="bash",
            type="bash_20250124",
            strict=True,
        )

    You can do:
        from langchain_anthropic import BetaToolBash20250124Param
        tool = BetaToolBash20250124Param(strict=True)
"""

from collections.abc import Callable
from typing import Any, TypeVar

from anthropic.types.beta import (
    BetaCodeExecutionTool20250522Param as _BetaCodeExecutionTool20250522Param,
)
from anthropic.types.beta import (
    BetaCodeExecutionTool20250825Param as _BetaCodeExecutionTool20250825Param,
)
from anthropic.types.beta import (
    BetaToolBash20241022Param as _BetaToolBash20241022Param,
)
from anthropic.types.beta import (
    BetaToolBash20250124Param as _BetaToolBash20250124Param,
)
from anthropic.types.beta import (
    BetaToolComputerUse20241022Param as _BetaToolComputerUse20241022Param,
)
from anthropic.types.beta import (
    BetaToolComputerUse20250124Param as _BetaToolComputerUse20250124Param,
)
from anthropic.types.beta import (
    BetaToolTextEditor20241022Param as _BetaToolTextEditor20241022Param,
)
from anthropic.types.beta import (
    BetaToolTextEditor20250124Param as _BetaToolTextEditor20250124Param,
)
from anthropic.types.beta import (
    BetaToolTextEditor20250429Param as _BetaToolTextEditor20250429Param,
)
from anthropic.types.beta import (
    BetaWebFetchTool20250910Param as _BetaWebFetchTool20250910Param,
)
from anthropic.types.beta import (
    BetaWebSearchTool20250305Param as _BetaWebSearchTool20250305Param,
)

T = TypeVar("T")


def _make_tool_wrapper(cls: type[T], literals: dict[str, Any]) -> Callable[..., T]:
    """Create a wrapper function that auto-populates literal fields.

    Args:
        cls: The TypedDict class to wrap.
        literals: Dict of field names to their literal values.

    Returns:
        A callable that returns the TypedDict with literals pre-filled.
    """

    def wrapper(**kwargs: Any) -> T:
        return {**literals, **kwargs}  # type: ignore[return-value]

    wrapper.__doc__ = f"Create a {cls.__name__} with auto-populated literal fields."
    wrapper.__annotations__["return"] = cls
    wrapper.__name__ = cls.__name__
    wrapper.__qualname__ = cls.__name__
    return wrapper


# Bash tools
BetaToolBash20241022Param = _make_tool_wrapper(
    _BetaToolBash20241022Param,
    {"name": "bash", "type": "bash_20241022"},
)

BetaToolBash20250124Param = _make_tool_wrapper(
    _BetaToolBash20250124Param,
    {"name": "bash", "type": "bash_20250124"},
)

# Text editor tools
BetaToolTextEditor20241022Param = _make_tool_wrapper(
    _BetaToolTextEditor20241022Param,
    {"name": "str_replace_editor", "type": "text_editor_20241022"},
)

BetaToolTextEditor20250124Param = _make_tool_wrapper(
    _BetaToolTextEditor20250124Param,
    {"name": "str_replace_editor", "type": "text_editor_20250124"},
)

BetaToolTextEditor20250429Param = _make_tool_wrapper(
    _BetaToolTextEditor20250429Param,
    {"name": "str_replace_editor", "type": "text_editor_20250429"},
)

# Computer use tools (note: still require display_width_px and display_height_px)
BetaToolComputerUse20241022Param = _make_tool_wrapper(
    _BetaToolComputerUse20241022Param,
    {"name": "computer", "type": "computer_20241022"},
)

BetaToolComputerUse20250124Param = _make_tool_wrapper(
    _BetaToolComputerUse20250124Param,
    {"name": "computer", "type": "computer_20250124"},
)

# Code execution tools
BetaCodeExecutionTool20250522Param = _make_tool_wrapper(
    _BetaCodeExecutionTool20250522Param,
    {"name": "code_execution", "type": "code_execution_20250522"},
)

BetaCodeExecutionTool20250825Param = _make_tool_wrapper(
    _BetaCodeExecutionTool20250825Param,
    {"name": "code_execution", "type": "code_execution_20250825"},
)

# Web search tool
BetaWebSearchTool20250305Param = _make_tool_wrapper(
    _BetaWebSearchTool20250305Param,
    {"name": "web_search", "type": "web_search_20250305"},
)

# Web fetch tool
BetaWebFetchTool20250910Param = _make_tool_wrapper(
    _BetaWebFetchTool20250910Param,
    {"name": "web_fetch", "type": "web_fetch_20250910"},
)

__all__ = [
    "BetaCodeExecutionTool20250522Param",
    "BetaCodeExecutionTool20250825Param",
    "BetaToolBash20241022Param",
    "BetaToolBash20250124Param",
    "BetaToolComputerUse20241022Param",
    "BetaToolComputerUse20250124Param",
    "BetaToolTextEditor20241022Param",
    "BetaToolTextEditor20250124Param",
    "BetaToolTextEditor20250429Param",
    "BetaWebFetchTool20250910Param",
    "BetaWebSearchTool20250305Param",
]
