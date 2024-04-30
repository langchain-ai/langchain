from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.document_loaders.airbyte import (
        AirbyteCDKLoader,
        AirbyteGongLoader,
        AirbyteHubspotLoader,
        AirbyteSalesforceLoader,
        AirbyteShopifyLoader,
        AirbyteStripeLoader,
        AirbyteTypeformLoader,
        AirbyteZendeskSupportLoader,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "AirbyteCDKLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteHubspotLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteStripeLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteTypeformLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteZendeskSupportLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteShopifyLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteSalesforceLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteGongLoader": "langchain_community.document_loaders.airbyte",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "AirbyteCDKLoader",
    "AirbyteHubspotLoader",
    "AirbyteStripeLoader",
    "AirbyteTypeformLoader",
    "AirbyteZendeskSupportLoader",
    "AirbyteShopifyLoader",
    "AirbyteSalesforceLoader",
    "AirbyteGongLoader",
]
