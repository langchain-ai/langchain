from importlib import metadata

from langchain_notion.notion_wrapper import NotionWrapper
from langchain_notion.toolkits import LangchainNotionToolkit
from langchain_notion.tools import LangchainNotionTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "LangchainNotionToolkit",
    "LangchainNotionTool",
    "NotionWrapper",
    "__version__",
]
