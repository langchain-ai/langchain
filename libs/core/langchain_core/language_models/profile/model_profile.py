"""Model profile types and utilities."""

from typing import Any

from typing_extensions import TypedDict


class ModelProfile(TypedDict, total=False):
    """Model profile.

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


def map_raw_data_to_profile(data: dict[str, Any]) -> ModelProfile:
    """Map raw model data to ModelProfile format.

    This function is used by provider packages to convert raw data from models.dev
    and augmentations into the standardized ModelProfile format.

    Args:
        data: Raw model data from models.dev and augmentations.

    Returns:
        ModelProfile with standardized fields.
    """
    # Map models.dev & augmentation fields -> ModelProfile fields
    # See schema reference: https://github.com/sst/models.dev?tab=readme-ov-file#schema-reference
    profile = {
        "max_input_tokens": data.get("limit", {}).get("context"),
        "image_inputs": "image" in data.get("modalities", {}).get("input", []),
        "image_url_inputs": data.get("image_url_inputs"),
        "image_tool_message": data.get("image_tool_message"),
        "audio_inputs": "audio" in data.get("modalities", {}).get("input", []),
        "pdf_inputs": "pdf" in data.get("modalities", {}).get("input", [])
        or data.get("pdf_inputs"),
        "pdf_tool_message": data.get("pdf_tool_message"),
        "video_inputs": "video" in data.get("modalities", {}).get("input", []),
        "max_output_tokens": data.get("limit", {}).get("output"),
        "reasoning_output": data.get("reasoning"),
        "image_outputs": "image" in data.get("modalities", {}).get("output", []),
        "audio_outputs": "audio" in data.get("modalities", {}).get("output", []),
        "video_outputs": "video" in data.get("modalities", {}).get("output", []),
        "tool_calling": data.get("tool_call"),
        "tool_choice": data.get("tool_choice"),
        "structured_output": data.get("structured_output"),
    }

    return ModelProfile(**{k: v for k, v in profile.items() if v is not None})  # type: ignore[typeddict-item]
