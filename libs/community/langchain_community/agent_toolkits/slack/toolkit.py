from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.slack.get_channel import SlackGetChannel
from langchain_community.tools.slack.get_message import SlackGetMessage
from langchain_community.tools.slack.schedule_message import SlackScheduleMessage
from langchain_community.tools.slack.send_message import SlackSendMessage
from langchain_community.tools.slack.utils import login

if TYPE_CHECKING:
    from slack_sdk import WebClient


class SlackToolkit(BaseToolkit):
    """Toolkit for interacting with Slack.

    Parameters:
        client: The Slack client.
    """

    client: WebClient = Field(default_factory=login)

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            SlackGetChannel(),
            SlackGetMessage(),
            SlackScheduleMessage(),
            SlackSendMessage(),
        ]
