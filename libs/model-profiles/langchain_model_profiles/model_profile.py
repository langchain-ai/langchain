"""Model profiles package."""

from typing_extensions import TypedDict

from langchain_model_profiles._data_loader import _DataLoader


class ModelProfile(TypedDict, total=False):
    """Model profile."""

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
    """TODO: description."""

    pdf_tool_message: bool
    """TODO: description."""

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
    """Whether the model supports [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs)"""


_LOADER = _DataLoader()

_lc_type_to_provider_id = {
    "openai-chat": "openai",
    "anthropic-chat": "anthropic",
}


def get_model_profile(provider_id: str, model_id: str) -> ModelProfile | None:
    """Get the model capabilities for a given model.

    Args:
        provider_id: Identifier for provider (e.g., `'openai'`, `'anthropic'`).
        model_id: Identifier for model (e.g., `'gpt-5'`,
            `'claude-sonnet-4-5-20250929'`).

    Returns:
        The model capabilities or `None` if not found in the data.
    """
    if not provider_id or not model_id:
        return None

    data = _LOADER.get_profile_data(
        _lc_type_to_provider_id.get(provider_id, provider_id), model_id
    )
    if not data:
        # If either (1) provider not found or (2) model not found under matched provider
        return None

    # Map models.dev & augmentation fields -> ModelProfile fields
    # See schema reference to see fields dropped: https://github.com/sst/models.dev?tab=readme-ov-file#schema-reference
    profile = {
        "max_input_tokens": data.get("limit", {}).get("context"),
        "image_inputs": "image" in data.get("modalities", {}).get("input", []),
        "image_url_inputs": data.get("image_url_inputs"),
        "image_tool_message": data.get("image_tool_message"),
        "audio_inputs": "audio" in data.get("modalities", {}).get("input", []),
        "pdf_inputs": "pdf" in data.get("modalities", {}).get("input", []),
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
