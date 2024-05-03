"""Slack tools."""
from typing import TYPE_CHECKING, Any

from langchain._api import create_importer

if TYPE_CHECKING:
    from langchain_community.tools import (
        SlackGetChannel,
        SlackGetMessage,
        SlackScheduleMessage,
        SlackSendMessage,
    )

# Create a way to dynamically look up deprecated imports.
# Used to consolidate logic for raising deprecation warnings and
# handling optional imports.
DEPRECATED_LOOKUP = {
    "SlackGetChannel": "langchain_community.tools",
    "SlackGetMessage": "langchain_community.tools",
    "SlackScheduleMessage": "langchain_community.tools",
    "SlackSendMessage": "langchain_community.tools",
}

_import_attribute = create_importer(__package__, deprecated_lookups=DEPRECATED_LOOKUP)


def __getattr__(name: str) -> Any:
    """Look up attributes dynamically."""
    return _import_attribute(name)


__all__ = [
    "SlackGetChannel",
    "SlackGetMessage",
    "SlackScheduleMessage",
    "SlackSendMessage",
]
