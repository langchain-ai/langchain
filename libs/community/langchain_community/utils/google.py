"""Utilities to use Google provided components."""

from importlib import metadata
from typing import Any, Optional


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
