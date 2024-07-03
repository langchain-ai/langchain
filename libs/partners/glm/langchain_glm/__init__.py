# ruff: noqa: E402
"""Main entrypoint into package."""
from importlib import metadata

from langchain_glm.agents import ZhipuAIAllToolsRunnable
from langchain_glm.chat_models import ChatZhipuAI

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)


__all__ = [
    "ChatZhipuAI",
    "ZhipuAIAllToolsRunnable",
]
