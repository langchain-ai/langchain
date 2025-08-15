from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.utilities.anthropic import (
        get_num_tokens_anthropic,
        get_token_ids_anthropic,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "get_num_tokens_anthropic": "langchain_community.utilities.anthropic",
    "get_token_ids_anthropic": "langchain_community.utilities.anthropic",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "get_num_tokens_anthropic",
    "get_token_ids_anthropic",
]
