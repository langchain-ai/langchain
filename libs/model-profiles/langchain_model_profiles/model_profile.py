"""Model profiles package."""

import re

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
    """Whether the model supports a native [structured output](https://docs.langchain.com/oss/python/langchain/models#structured-outputs)
    feature"""


_LOADER = _DataLoader()

_lc_type_to_provider_id = {
    "openai-chat": "openai",
    "azure-openai-chat": "azure",
    "anthropic-chat": "anthropic",
    "chat-google-generative-ai": "google",
    "vertexai": "google-vertex",
    "anthropic-chat-vertexai": "google-vertex-anthropic",
    "amazon_bedrock_chat": "amazon-bedrock",
    "amazon_bedrock_converse_chat": "amazon-bedrock",
    "chat-ai21": "ai21",
    "chat-deepseek": "deepseek",
    "fireworks-chat": "fireworks-ai",
    "groq-chat": "groq",
    "huggingface-chat-wrapper": "huggingface",
    "mistralai-chat": "mistral",
    "chat-ollama": "ollama",
    "perplexitychat": "perplexity",
    "together-chat": "togetherai",
    "upstage-chat": "upstage",
    "xai-chat": "xai",
}


def _translate_provider_and_model_id(provider: str, model: str) -> tuple[str, str]:
    """Translate LangChain provider and model to models.dev equivalents.

    Args:
        provider: LangChain provider ID.
        model: LangChain model ID.

    Returns:
        A tuple containing the models.dev provider ID and model ID.
    """
    provider_id = _lc_type_to_provider_id.get(provider, provider)

    if provider_id in ("google", "google-vertex"):
        # convert models/gemini-2.0-flash-001 to gemini-2.0-flash
        model_id = re.sub(r"-\d{3}$", "", model.replace("models/", ""))
    elif provider_id == "amazon-bedrock":
        # strip region prefixes like "us."
        model_id = re.sub(r"^[A-Za-z]{2}\.", "", model)
    else:
        model_id = model

    return provider_id, model_id


def get_model_profile(provider: str, model: str) -> ModelProfile | None:
    """Get the model capabilities for a given model.

    Args:
        provider: Identifier for provider (e.g., `'openai'`, `'anthropic'`).
        model: Identifier for model (e.g., `'gpt-5'`,
            `'claude-sonnet-4-5-20250929'`).

    Returns:
        The model capabilities or `None` if not found in the data.
    """
    if not provider or not model:
        return None

    provider_id, model_id = _translate_provider_and_model_id(provider, model)
    data = _LOADER.get_profile_data(provider_id, model_id)
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
