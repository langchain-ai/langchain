from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.office365.create_draft_message import (
    O365CreateDraftMessage,
)
from langchain_community.tools.office365.events_search import O365SearchEvents
from langchain_community.tools.office365.messages_search import O365SearchEmails
from langchain_community.tools.office365.send_event import O365SendEvent
from langchain_community.tools.office365.send_message import O365SendMessage
from langchain_community.tools.office365.utils import authenticate

if TYPE_CHECKING:
    from O365 import Account


class O365Toolkit(BaseToolkit):
    """Toolkit for interacting with Office 365.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by reading, creating, updating, deleting
        data associated with this service.

        For example, this toolkit can be used search through emails and events,
        send messages and event invites, and create draft messages.

        Please make sure that the permissions given by this toolkit
        are appropriate for your use case.

        See https://python.langchain.com/docs/security for more information.

    Parameters:
        account: Optional. The Office 365 account. Default is None.
    """

    account: Account = Field(default_factory=authenticate)

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            O365SearchEvents(),
            O365CreateDraftMessage(),
            O365SearchEmails(),
            O365SendEvent(),
            O365SendMessage(),
        ]
