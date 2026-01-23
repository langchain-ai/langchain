"""LangChain integrations for Meshtastic LoRa mesh networks.

This package provides tools for sending messages via Meshtastic devices,
enabling AI agents to communicate over decentralized mesh networks.
"""

from importlib import metadata
from importlib.metadata import PackageNotFoundError

from langchain_meshtastic.tools import MeshtasticSendInput, MeshtasticSendTool


def _raise_package_not_found_error() -> None:
    raise PackageNotFoundError


try:
    if __package__ is None:
        _raise_package_not_found_error()
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata

__all__ = [
    "MeshtasticSendInput",
    "MeshtasticSendTool",
    "__version__",
]
