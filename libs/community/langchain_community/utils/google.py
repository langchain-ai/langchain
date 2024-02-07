"""Utilities to use Google provided components."""

from importlib import metadata
from typing import Any, Callable, Optional, Union

import google.api_core
from google.api_core.gapic_v1.client_info import ClientInfo
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.llms import create_base_retry_decorator


def create_retry_decorator(
    *,
    max_retries: int = 1,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Creates a retry decorator for Vertex / Palm LLMs."""

    errors = [
        google.api_core.exceptions.ResourceExhausted,
        google.api_core.exceptions.ServiceUnavailable,
        google.api_core.exceptions.Aborted,
        google.api_core.exceptions.DeadlineExceeded,
        google.api_core.exceptions.GoogleAPIError,
    ]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=run_manager
    )
    return decorator


def get_client_info(module: Optional[str] = None) -> "ClientInfo":
    r"""Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo
    """
    langchain_version = metadata.version("langchain")
    client_library_version = (
        f"{langchain_version}-{module}" if module else langchain_version
    )
    return ClientInfo(
        client_library_version=client_library_version,
        user_agent=f"langchain/{client_library_version}",
    )
