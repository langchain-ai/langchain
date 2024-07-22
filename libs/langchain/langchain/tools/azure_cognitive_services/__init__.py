"""Azure Cognitive Services Tools."""

from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        AzureCogsFormRecognizerTool,
        AzureCogsImageAnalysisTool,
        AzureCogsSpeech2TextTool,
        AzureCogsText2SpeechTool,
        AzureCogsTextAnalyticsHealthTool,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "AzureCogsImageAnalysisTool": "langchain_community.tools",
    "AzureCogsFormRecognizerTool": "langchain_community.tools",
    "AzureCogsSpeech2TextTool": "langchain_community.tools",
    "AzureCogsText2SpeechTool": "langchain_community.tools",
    "AzureCogsTextAnalyticsHealthTool": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AzureCogsImageAnalysisTool",
    "AzureCogsFormRecognizerTool",
    "AzureCogsSpeech2TextTool",
    "AzureCogsText2SpeechTool",
    "AzureCogsTextAnalyticsHealthTool",
]
