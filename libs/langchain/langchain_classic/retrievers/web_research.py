from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.retrievers.web_research import (
        QuestionListOutputParser,
        SearchQueries,
        WebResearchRetriever,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "QuestionListOutputParser": "langchain_community.retrievers.web_research",
    "SearchQueries": "langchain_community.retrievers.web_research",
    "WebResearchRetriever": "langchain_community.retrievers.web_research",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = ["QuestionListOutputParser", "SearchQueries", "WebResearchRetriever"]
