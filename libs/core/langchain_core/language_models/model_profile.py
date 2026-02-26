"""Model profile types and utilities."""

from typing_extensions import TypedDict


class ModelProfile(TypedDict, total=False):
    """Model profile.

    !!! warning "Beta feature"
        This is a beta feature. The format of model profiles is subject to change.

    Provides information about chat model capabilities, such as context window sizes
    and supported features.
    """

    # --- Input constraints ---

    max_input_tokens: int
    """Maximum context window (tokens)"""

    image_inputs: bool
    """Whether image inputs are supported."""
    # TODO: add more detail about formats?

    image_url_inputs: bool
    """Whether [image URL inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported."""

    pdf_inputs: bool
    """Whether [PDF inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported."""
    # TODO: add more detail about formats? e.g. bytes or base64

    audio_inputs: bool
    """Whether [audio inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported."""
    # TODO: add more detail about formats? e.g. bytes or base64

    video_inputs: bool
    """Whether [video inputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported."""
    # TODO: add more detail about formats? e.g. bytes or base64

    image_tool_message: bool
    """Whether images can be included in tool messages."""

    pdf_tool_message: bool
    """Whether PDFs can be included in tool messages."""

    # --- Output constraints ---

    max_output_tokens: int
    """Maximum output tokens"""

    reasoning_output: bool
    """Whether the model supports [reasoning / chain-of-thought](https://docs.langchain.com/oss/python/langchain/models#reasoning)"""

    image_outputs: bool
    """Whether [image outputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported."""

    audio_outputs: bool
    """Whether [audio outputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported."""

    video_outputs: bool
    """Whether [video outputs](https://docs.langchain.com/oss/python/langchain/models#multimodal)
    are supported."""

    # --- Tool calling ---
    tool_calling: bool
    """Whether the model supports [tool calling](https://docs.langchain.com/oss/python/langchain/models#tool-calling)"""

    tool_choice: bool
    """Whether the model supports [tool choice](https://docs.langchain.com/oss/python/langchain/models#forcing-tool-calls)"""

    # --- Structured output ---
    structured_output: bool
    """Whether the model supports a native [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs)
    feature"""


ModelProfileRegistry = dict[str, ModelProfile]
"""Registry mapping model identifiers or names to their ModelProfile."""
