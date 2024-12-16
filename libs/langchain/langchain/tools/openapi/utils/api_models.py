from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import APIOperation
    from langchain_community.tools.openapi.utils.api_models import (
        INVALID_LOCATION_TEMPL,
        PRIMITIVE_TYPES,
        SCHEMA_TYPE,
        SUPPORTED_LOCATIONS,
        APIProperty,
        APIPropertyBase,
        APIPropertyLocation,
        APIRequestBody,
        APIRequestBodyProperty,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "APIPropertyLocation": "langchain_community.tools.openapi.utils.api_models",
    "APIPropertyBase": "langchain_community.tools.openapi.utils.api_models",
    "APIProperty": "langchain_community.tools.openapi.utils.api_models",
    "APIRequestBodyProperty": "langchain_community.tools.openapi.utils.api_models",
    "APIRequestBody": "langchain_community.tools.openapi.utils.api_models",
    "APIOperation": "langchain_community.tools",
    "INVALID_LOCATION_TEMPL": "langchain_community.tools.openapi.utils.api_models",
    "SCHEMA_TYPE": "langchain_community.tools.openapi.utils.api_models",
    "PRIMITIVE_TYPES": "langchain_community.tools.openapi.utils.api_models",
    "SUPPORTED_LOCATIONS": "langchain_community.tools.openapi.utils.api_models",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "PRIMITIVE_TYPES",
    "APIPropertyLocation",
    "SUPPORTED_LOCATIONS",
    "INVALID_LOCATION_TEMPL",
    "SCHEMA_TYPE",
    "APIPropertyBase",
    "APIProperty",
    "APIRequestBodyProperty",
    "APIRequestBody",
    "APIOperation",
]
