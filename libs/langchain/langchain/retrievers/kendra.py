from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.retrievers import AmazonKendraRetriever
    from langchain_community.retrievers.kendra import (
        AdditionalResultAttribute,
        AdditionalResultAttributeValue,
        DocumentAttribute,
        DocumentAttributeValue,
        Highlight,
        QueryResult,
        QueryResultItem,
        ResultItem,
        RetrieveResult,
        RetrieveResultItem,
        TextWithHighLights,
        clean_excerpt,
        combined_text,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "clean_excerpt": "langchain_community.retrievers.kendra",
    "combined_text": "langchain_community.retrievers.kendra",
    "Highlight": "langchain_community.retrievers.kendra",
    "TextWithHighLights": "langchain_community.retrievers.kendra",
    "AdditionalResultAttributeValue": "langchain_community.retrievers.kendra",
    "AdditionalResultAttribute": "langchain_community.retrievers.kendra",
    "DocumentAttributeValue": "langchain_community.retrievers.kendra",
    "DocumentAttribute": "langchain_community.retrievers.kendra",
    "ResultItem": "langchain_community.retrievers.kendra",
    "QueryResultItem": "langchain_community.retrievers.kendra",
    "RetrieveResultItem": "langchain_community.retrievers.kendra",
    "QueryResult": "langchain_community.retrievers.kendra",
    "RetrieveResult": "langchain_community.retrievers.kendra",
    "AmazonKendraRetriever": "langchain_community.retrievers",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "clean_excerpt",
    "combined_text",
    "Highlight",
    "TextWithHighLights",
    "AdditionalResultAttributeValue",
    "AdditionalResultAttribute",
    "DocumentAttributeValue",
    "DocumentAttribute",
    "ResultItem",
    "QueryResultItem",
    "RetrieveResultItem",
    "QueryResult",
    "RetrieveResult",
    "AmazonKendraRetriever",
]
