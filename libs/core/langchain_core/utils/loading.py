"""Utilities for loading configurations from langchain_core-hub."""

import warnings
from typing import Any

from langchain_core._api.deprecation import deprecated


@deprecated(
    since="0.1.30",
    removal="1.0",
    message=(
        "Using the hwchase17/langchain-hub "
        "repo for prompts is deprecated. Please use "
        "<https://smith.langchain.com/hub> instead."
    ),
)
def try_load_from_hub(
    *args: Any,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """[DEPRECATED] Try to load from the old Hub.

    Returns:
        None always, indicating that we shouldn't load from the old hub.
    """
    warnings.warn(
        "Loading from the deprecated github-based Hub is no longer supported. "
        "Please use the new LangChain Hub at https://smith.langchain.com/hub instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # return None, which indicates that we shouldn't load from old hub
    # and might just be a filepath for e.g. load_chain
    return None
