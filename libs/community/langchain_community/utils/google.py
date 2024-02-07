"""Utilities to use Google provided components."""

from importlib import metadata
from typing import Any, Callable, Optional, Union

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
    from google.api_core.exceptions import (
        Aborted,
        DeadlineExceeded,
        GoogleAPIError,
        ResourceExhausted,
        ServiceUnavailable,
    )

    errors = [
        ResourceExhausted,
        ServiceUnavailable,
        Aborted,
        DeadlineExceeded,
        GoogleAPIError,
    ]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=max_retries, run_manager=run_manager
    )
    return decorator


def get_client_info(module: Optional[str] = None) -> Any:
    r"""Returns a custom user agent header.

    Args:
        module (Optional[str]):
            Optional. The module for a custom user agent header.
    Returns:
        google.api_core.gapic_v1.client_info.ClientInfo
    """
    from google.api_core.gapic_v1.client_info import ClientInfo

    langchain_version = metadata.version("langchain")
    client_library_version = (
        f"{langchain_version}-{module}" if module else langchain_version
    )
    return ClientInfo(
        client_library_version=client_library_version,
        user_agent=f"langchain/{client_library_version}",
    )
