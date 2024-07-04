"""For backwards compatibility."""
from typing import Any

from langchain._api import create_importer

# Code has been removed from the community package as well.
# We'll proxy to community package, which will raise an appropriate exception,
# but we'll not include this in __all__, so it won't be listed as importable.

_importer = create_importer(
    __package__,
    deprecated_lookups={"PythonREPL": "langchain_community.utilities.python"},
)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _importer(name)
