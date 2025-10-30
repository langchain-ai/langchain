"""Model profiles package."""

from typing_extensions import TypedDict

from langchain_model_profiles._data_loader import _DataLoader


class ModelProfile(TypedDict, total=False):
    """Model profile."""

    # --- Input constraints ---
    max_input_tokens: int
    image_inputs: bool
    image_url_inputs: bool
    pdf_inputs: bool
    audio_inputs: bool
    video_inputs: bool
    image_tool_message: bool
    pdf_tool_message: bool

    # --- Output constraints ---
    max_output_tokens: int
    reasoning_output: bool
    image_outputs: bool
    audio_outputs: bool
    video_outputs: bool

    # --- Tool calling ---
    tool_calling: bool
    tool_choice: bool

    # --- Structured output ---
    structured_output: bool


_loader = _DataLoader()

_lc_type_to_provider_id = {
    "openai-chat": "openai",
    "anthropic-chat": "anthropic",
}


def get_model_profile(provider_id: str, model_id: str) -> ModelProfile | None:
    """Get the model capabilities for a given model.

    Args:
        provider_id: Identifier for provider (e.g., `'openai'`, `'anthropic'`).
        model_id: Identifier for model (e.g., `'gpt-5'`, `'claude-sonnet-4-5-20250929'`).

    Returns:
        The model capabilities or `None` if not found in the data.
    """
    if not provider_id or not model_id:
        return None

    data = _loader.get_profile_data(
        _lc_type_to_provider_id.get(provider_id, provider_id), model_id
    )
    if not data:
        return None

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
