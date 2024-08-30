"""Different methods for rendering Tools to be passed to LLMs.

Depending on the LLM you are using and the prompting strategy you are using,
you may want Tools to be rendered in a different way.
This module contains various ways to render tools.
"""

# For backwards compatibility
from langchain_core.tools import (
    render_text_description,
    render_text_description_and_args,
)
from langchain_core.utils.function_calling import (
    format_tool_to_openai_function,
    format_tool_to_openai_tool,
)

__all__ = [
    "render_text_description",
    "render_text_description_and_args",
    "format_tool_to_openai_tool",
    "format_tool_to_openai_function",
]
