"""For backwards compatibility."""
from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utilities.python import PythonREPL


_importer = create_importer(
    __package__,
    deprecated_lookups={"PythonREPL": "langchain_community.utilities.python"},
)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _importer(name)


__all__ = ["PythonREPL"]
