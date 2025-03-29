from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_azure_ai.chat_models import AzureAIChatCompletionsModel

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "AzureChatOpenAI": "langchain_community.chat_models.azure_openai",
    "AzureAIChatCompletionsModel": "langchain_azure_ai.chat_models",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AzureChatOpenAI",
    "AzureAIChatCompletionsModel",
]
