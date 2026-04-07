"""Model profile types and utilities."""

import logging
import warnings
from typing import get_type_hints

from pydantic import ConfigDict
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class ModelProfile(TypedDict, total=False):
    """Model profile.

    !!! warning "Beta feature"

        This is a beta feature. The format of model profiles is subject to change.

    Provides information about chat model capabilities, such as context window sizes
    and supported features.
    """

    __pydantic_config__ = ConfigDict(extra="allow")  # type: ignore[misc]

    # --- Model metadata ---

    name: str
    """Human-readable model name."""

    status: str
    """Model status (e.g., `'active'`, `'deprecated'`)."""

    release_date: str
    """Model release date (ISO 8601 format, e.g., `'2025-06-01'`)."""

    last_updated: str
    """Date the model was last updated (ISO 8601 format)."""

    open_weights: bool
    """Whether the model weights are openly available."""

    # --- Input constraints ---

    max_input_tokens: int
    """Maximum context window (tokens)"""

    text_inputs: bool
    """Whether text inputs are supported."""

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

    text_outputs: bool
    """Whether text outputs are supported."""

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

    # --- Other capabilities ---

    attachment: bool
    """Whether the model supports file attachments."""

    temperature: bool
    """Whether the model supports a temperature parameter."""


ModelProfileRegistry = dict[str, ModelProfile]
"""Registry mapping model identifiers or names to their ModelProfile."""


def _warn_unknown_profile_keys(profile: ModelProfile) -> None:
    """Warn if `profile` contains keys not declared on `ModelProfile`.

    Args:
        profile: The model profile dict to check for undeclared keys.
    """
    if not isinstance(profile, dict):
        return

    try:
        declared = frozenset(get_type_hints(ModelProfile).keys())
    except (TypeError, NameError):
        # get_type_hints raises NameError on unresolvable forward refs and
        # TypeError when annotations evaluate to non-type objects.
        logger.debug(
            "Could not resolve type hints for ModelProfile; "
            "skipping unknown-key check.",
            exc_info=True,
        )
        return

    extra = sorted(set(profile) - declared)
    if extra:
        warnings.warn(
            f"Unrecognized keys in model profile: {extra}. "
            f"This may indicate a version mismatch between langchain-core "
            f"and your provider package. Consider upgrading langchain-core.",
            stacklevel=2,
        )
