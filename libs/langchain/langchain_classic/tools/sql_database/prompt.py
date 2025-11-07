"""For backwards compatibility."""

from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools.sql_database.prompt import QUERY_CHECKER


_importer = create_importer(
    __package__,
    deprecated_lookups={
        "QUERY_CHECKER": "langchain_community.tools.sql_database.prompt",
    },
)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _importer(name)


__all__ = ["QUERY_CHECKER"]
