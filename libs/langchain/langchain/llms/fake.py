from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.llms.fake import FakeStreamingListLLM
    from langchain_core.language_models import FakeListLLM

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "FakeListLLM": "langchain_community.llms",
    "FakeStreamingListLLM": "langchain_community.llms.fake",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "FakeListLLM",
    "FakeStreamingListLLM",
]
