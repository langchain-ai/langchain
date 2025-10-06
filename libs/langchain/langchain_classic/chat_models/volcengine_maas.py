from typing import TYPE_CHECKING, Any

from langchain_classic._api import create_importer

if TYPE_CHECKING:
    from langchain_community.chat_models.volcengine_maas import (
        VolcEngineMaasChat,
        convert_dict_to_message,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "convert_dict_to_message": "langchain_community.chat_models.volcengine_maas",
    "VolcEngineMaasChat": "langchain_community.chat_models.volcengine_maas",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "VolcEngineMaasChat",
    "convert_dict_to_message",
]
