from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

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

    Setup:
        Install ``slack_sdk`` and set environment variable ``SLACK_USER_TOKEN``.

        .. code-block:: bash

            pip install -U slack_sdk
            export SLACK_USER_TOKEN="your-user-token"

    Key init args:
        client: slack_sdk.WebClient
            The Slack client.

    Instantiate:
        .. code-block:: python

            from langchain_community.agent_toolkits import SlackToolkit

            toolkit = SlackToolkit()

    Tools:
        .. code-block:: python

            tools = toolkit.get_tools()
            tools

        .. code-block:: none

            [SlackGetChannel(client=<slack_sdk.web.client.WebClient object at 0x113caa8c0>),
            SlackGetMessage(client=<slack_sdk.web.client.WebClient object at 0x113caa4d0>),
            SlackScheduleMessage(client=<slack_sdk.web.client.WebClient object at 0x113caa440>),
            SlackSendMessage(client=<slack_sdk.web.client.WebClient object at 0x113caa410>)]

    Use within an agent:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(model="gpt-4o-mini")
            agent_executor = create_react_agent(llm, tools)

            example_query = "When was the #general channel created?"

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                message = event["messages"][-1]
                if message.type != "tool":  # mask sensitive information
                    event["messages"][-1].pretty_print()

        .. code-block:: none

             ================================[1m Human Message [0m=================================

            When was the #general channel created?
            ==================================[1m Ai Message [0m==================================
            Tool Calls:
            get_channelid_name_dict (call_NXDkALjoOx97uF1v0CoZTqtJ)
            Call ID: call_NXDkALjoOx97uF1v0CoZTqtJ
            Args:
            ==================================[1m Ai Message [0m==================================

            The #general channel was created on timestamp 1671043305.
    """  # noqa: E501

    client: WebClient = Field(default_factory=login)

    class Config:
        arbitrary_types_allowed = True

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return [
            SlackGetChannel(),
            SlackGetMessage(),
            SlackScheduleMessage(),
            SlackSendMessage(),
        ]
