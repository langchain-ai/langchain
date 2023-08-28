from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_xfyun.agents.agent_toolkits.base import BaseToolkit
from langchain_xfyun.pydantic_v1 import Field
from langchain_xfyun.tools import BaseTool
from langchain_xfyun.tools.gmail.create_draft import GmailCreateDraft
from langchain_xfyun.tools.gmail.get_message import GmailGetMessage
from langchain_xfyun.tools.gmail.get_thread import GmailGetThread
from langchain_xfyun.tools.gmail.search import GmailSearch
from langchain_xfyun.tools.gmail.send_message import GmailSendMessage
from langchain_xfyun.tools.gmail.utils import build_resource_service

if TYPE_CHECKING:
    # This is for linting and IDE typehints
    from googleapiclient.discovery import Resource
else:
    try:
        # We do this so pydantic can resolve the types when instantiating
        from googleapiclient.discovery import Resource
    except ImportError:
        pass


SCOPES = ["https://mail.google.com/"]


class GmailToolkit(BaseToolkit):
    """Toolkit for interacting with Gmail."""

    api_resource: Resource = Field(default_factory=build_resource_service)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            GmailCreateDraft(api_resource=self.api_resource),
            GmailSendMessage(api_resource=self.api_resource),
            GmailSearch(api_resource=self.api_resource),
            GmailGetMessage(api_resource=self.api_resource),
            GmailGetThread(api_resource=self.api_resource),
        ]
