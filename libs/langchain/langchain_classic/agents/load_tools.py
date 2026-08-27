from typing import Any

from langchain_classic._api import create_importer

_importer = create_importer(
    __package__,
    fallback_module="langchain_community.agent_toolkits.load_tools",
)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _importer(name)
