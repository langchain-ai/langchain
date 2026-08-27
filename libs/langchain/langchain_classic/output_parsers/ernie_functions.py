from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.output_parsers.ernie_functions import (
        JsonKeyOutputFunctionsParser,
        JsonOutputFunctionsParser,
        OutputFunctionsParser,
        PydanticAttrOutputFunctionsParser,
        PydanticOutputFunctionsParser,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "JsonKeyOutputFunctionsParser": (
        "langchain_community.output_parsers.ernie_functions"
    ),
    "JsonOutputFunctionsParser": "langchain_community.output_parsers.ernie_functions",
    "OutputFunctionsParser": "langchain_community.output_parsers.ernie_functions",
    "PydanticAttrOutputFunctionsParser": (
        "langchain_community.output_parsers.ernie_functions"
    ),
    "PydanticOutputFunctionsParser": (
        "langchain_community.output_parsers.ernie_functions"
    ),
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "JsonKeyOutputFunctionsParser",
    "JsonOutputFunctionsParser",
    "OutputFunctionsParser",
    "PydanticAttrOutputFunctionsParser",
    "PydanticOutputFunctionsParser",
]
