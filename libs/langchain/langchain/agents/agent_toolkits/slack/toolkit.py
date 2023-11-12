from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.pydantic_v1 import Field
from langchain.tools import BaseTool
from langchain.tools.slack.base import SlackBaseTool
from langchain.tools.slack.send_message import SlackSendMessage
from langchain.tools.slack.utils import authenticate
from slack_sdk import WebClient

if TYPE_CHECKING:
    # from O365 import Account

    from slack_sdk.errors import SlackApiError


class SlackToolkit(BaseToolkit):
    """Toolkit for interacting with Slack."""

    client: WebClient=Field(default_factory=authenticate)


    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[SlackBaseTool]:
        """Get the tools in the toolkit."""
        return [
            SlackSendMessage(client=self.client),
            # SlackSendMessage(),
        ]
