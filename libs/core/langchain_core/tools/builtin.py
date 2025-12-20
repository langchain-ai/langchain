"""Standard TypedDict schemas for builtin/server-side tools.

This module defines provider-agnostic schemas for common builtin tools like
web search, code execution, etc. Provider integrations (OpenAI, Anthropic, etc.)
use conversion utilities to translate these standard types to their provider-specific
formats.
"""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from typing_extensions import NotRequired, Required

__all__ = [
    "BashTool",
    "CodeExecutionTool",
    "FileSearchTool",
    "ImageGenerationTool",
    "MemoryTool",
    "TextEditorTool",
    "UserLocation",
    "WebFetchTool",
    "WebSearchTool",
    "XSearchTool",
]


class UserLocation(TypedDict, total=False):
    """User location for contextual search results.

    Used to provide more relevant search results based on user's location.
    Currently supported by OpenAI and Anthropic.
    """

    type: Required[Literal["approximate"]]
    """The type of location specification."""

    city: NotRequired[str]
    """The city of the user."""

    country: NotRequired[str]
    """The two letter ISO 3166-1 alpha-2 country code of the user."""

    region: NotRequired[str]
    """The region of the user."""

    timezone: NotRequired[str]
    """The IANA timezone of the user."""


class WebSearchTool(TypedDict, total=False):
    """Standard web search builtin tool.

    Supported by: OpenAI, Anthropic, xAI (Grok).

    Provider-specific notes:
    - Anthropic: Supports `max_uses` parameter
    - OpenAI: Supports `user_location` parameter
    - xAI: Basic support without additional parameters
    """

    type: Required[Literal["web_search"]]
    """The tool type identifier."""

    user_location: NotRequired[UserLocation]
    """User location for more relevant search results.

    Supported by OpenAI and Anthropic.
    """

    max_uses: NotRequired[int]
    """Maximum number of times the tool can be used in a single request.

    Anthropic-specific parameter.
    """


class CodeExecutionTool(TypedDict, total=False):
    """Standard code execution/interpreter builtin tool.

    Supported by: OpenAI, Anthropic, Google GenAI, xAI (Grok).

    Provider-specific notes:
    - OpenAI: Supports `container` parameter for execution environment
    - Anthropic: Uses code_execution_20250825 format
    - Google GenAI: Uses code_execution format
    - xAI: Uses code_interpreter format
    """

    type: Required[Literal["code_execution"]]
    """The tool type identifier."""

    container: NotRequired[dict[str, Any]]
    """Container configuration for code execution.

    OpenAI-specific parameter. Example: {"type": "auto"}
    """


class XSearchTool(TypedDict, total=False):
    """X/Twitter search tool.

    Supported by: xAI (Grok) only.

    Allows searching X/Twitter posts.
    """

    type: Required[Literal["x_search"]]
    """The tool type identifier."""


class WebFetchTool(TypedDict, total=False):
    """Web page fetching tool.

    Supported by: Anthropic only.

    Fetches and reads web pages.
    """

    type: Required[Literal["web_fetch"]]
    """The tool type identifier."""

    max_uses: NotRequired[int]
    """Maximum number of times the tool can be used in a single request.

    Anthropic-specific parameter.
    """


class MemoryTool(TypedDict, total=False):
    """Memory/context management tool.

    Supported by: Anthropic only.

    Allows the model to store and retrieve information across conversations.
    """

    type: Required[Literal["memory"]]
    """The tool type identifier."""


class FileSearchTool(TypedDict, total=False):
    """File search tool for searching through uploaded documents.

    Supported by: OpenAI only.

    Requires vector_store_ids to be configured.
    """

    type: Required[Literal["file_search"]]
    """The tool type identifier."""

    vector_store_ids: NotRequired[list[str]]
    """List of vector store IDs to search through.

    OpenAI-specific parameter. Required for file search to work.
    """


class ImageGenerationTool(TypedDict, total=False):
    """Image generation tool.

    Supported by: OpenAI only.

    Allows the model to generate images.
    """

    type: Required[Literal["image_generation"]]
    """The tool type identifier."""


class TextEditorTool(TypedDict, total=False):
    """Text editor tool for file editing.

    Supported by: Anthropic only.

    Provides string-replace-based editing capabilities.
    """

    type: Required[Literal["text_editor"]]
    """The tool type identifier."""


class BashTool(TypedDict, total=False):
    """Bash command execution tool.

    Supported by: Anthropic only.

    Allows execution of bash commands.
    """

    type: Required[Literal["bash"]]
    """The tool type identifier."""
