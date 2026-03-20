"""Model profile types and utilities."""

import contextlib
import warnings
from typing import get_type_hints

from pydantic import ConfigDict
from typing_extensions import TypedDict


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


# Cache for ModelProfile's declared field names. Populated lazily because
# _warn_unknown_profile_keys runs on every chat model construction and
# get_type_hints is not free.
_DECLARED_PROFILE_KEYS: frozenset[str] | None = None


def _get_declared_profile_keys() -> frozenset[str]:
    """Return the declared `ModelProfile` field names, cached after first call."""
    global _DECLARED_PROFILE_KEYS  # noqa: PLW0603
    if _DECLARED_PROFILE_KEYS is None:
        _DECLARED_PROFILE_KEYS = frozenset(get_type_hints(ModelProfile).keys())
    return _DECLARED_PROFILE_KEYS


def _warn_unknown_profile_keys(profile: ModelProfile) -> None:
    """Emit a warning if a profile dict contains keys not declared in `ModelProfile`.

    This function must never raise — it is called during model construction and
    a failure here would prevent all chat model instantiation.

    Args:
        profile: Model profile dict to check.
    """
    try:
        declared = _get_declared_profile_keys()
    except Exception:
        # If introspection fails (e.g. forward ref issues), skip rather than
        # crash model construction.
        return

    extra = sorted(set(profile) - declared)
    if extra:
        # warnings.warn() raises when the user (or a test framework like
        # pytest) configures warnings-as-errors (-W error /
        # warnings.simplefilter("error")). Suppress so we honour the
        # "must never raise" contract — this runs during every chat model
        # construction.
        with contextlib.suppress(Exception):
            warnings.warn(
                f"Unrecognized keys in model profile: {extra}. "
                f"This may indicate a version mismatch between langchain-core "
                f"and your provider package. Consider upgrading langchain-core.",
                stacklevel=2,
            )
