import os  # type: ignore[import-not-found]
import warnings
from typing import Any

from exa_py import AsyncExa, Exa
from langchain_core.utils import convert_to_secret_str

_CONTENT_KEYS = ("text", "highlights", "summary")


def build_contents_options(
    *,
    text: Any = None,
    highlights: Any = None,
    summary: Any = None,
    livecrawl: Any = None,
    max_age_hours: int | None = None,
) -> dict[str, Any]:
    """Build content options; defaults to text when none are set."""
    options = {
        "text": text,
        "highlights": highlights,
        "summary": summary,
        "livecrawl": livecrawl,
        "max_age_hours": max_age_hours,
    }
    selected = {key: value for key, value in options.items() if value is not None}
    if not any(key in selected for key in _CONTENT_KEYS):
        selected["text"] = True
    return selected


def warn_if_use_autoprompt(use_autoprompt: bool | None) -> None:  # noqa: FBT001
    """Warn that ``use_autoprompt`` is deprecated; use ``type="auto"``."""
    if use_autoprompt is not None:
        warnings.warn(
            "`use_autoprompt` is deprecated and no longer sent to Exa; "
            'use `type="auto"` instead.',
            DeprecationWarning,
            stacklevel=3,
        )


def initialize_client(values: dict) -> dict:
    """Initialize the client."""
    exa_api_key = values.get("exa_api_key") or os.environ.get("EXA_API_KEY") or ""
    values["exa_api_key"] = convert_to_secret_str(exa_api_key)
    if values.get("client") is not None and values.get("async_client") is not None:
        return values
    args = {
        "api_key": values["exa_api_key"].get_secret_value(),
    }
    if values.get("exa_base_url"):
        args["base_url"] = values["exa_base_url"]
    if values.get("client") is None:
        values["client"] = Exa(**args)
        values["client"].headers["x-exa-integration"] = (
            "langchain-ai/langchain-exa-integration"
        )
    if values.get("async_client") is None:
        async_args: dict[str, str] = {
            "api_key": values["exa_api_key"].get_secret_value()
        }
        if values.get("exa_base_url"):
            async_args["api_base"] = values["exa_base_url"]
        values["async_client"] = AsyncExa(**async_args)
        values["async_client"].headers["x-exa-integration"] = (
            "langchain-ai/langchain-exa-integration"
        )
    return values
