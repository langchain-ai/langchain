"""Edenai Tools."""

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        EdenAiExplicitImageTool,
        EdenAiObjectDetectionTool,
        EdenAiParsingIDTool,
        EdenAiParsingInvoiceTool,
        EdenAiSpeechToTextTool,
        EdenAiTextModerationTool,
        EdenAiTextToSpeechTool,
        EdenaiTool,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "EdenAiExplicitImageTool": "langchain_community.tools",
    "EdenAiObjectDetectionTool": "langchain_community.tools",
    "EdenAiParsingIDTool": "langchain_community.tools",
    "EdenAiParsingInvoiceTool": "langchain_community.tools",
    "EdenAiTextToSpeechTool": "langchain_community.tools",
    "EdenAiSpeechToTextTool": "langchain_community.tools",
    "EdenAiTextModerationTool": "langchain_community.tools",
    "EdenaiTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "EdenAiExplicitImageTool",
    "EdenAiObjectDetectionTool",
    "EdenAiParsingIDTool",
    "EdenAiParsingInvoiceTool",
    "EdenAiTextToSpeechTool",
    "EdenAiSpeechToTextTool",
    "EdenAiTextModerationTool",
    "EdenaiTool",
]
