"""Utilities for loading configurations from langchain_core-hub."""

from typing import Any


def try_load_from_hub(
    *args: Any,
    **kwargs: Any,
) -> Any:
    raise RuntimeError(
        "Loading from the deprecated github-based Hub is no longer supported. "
        "Please use the new LangChain Hub at https://smith.langchain.com/hub instead."
    )
