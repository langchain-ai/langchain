from __future__ import annotations

from typing import TYPE_CHECKING, List

from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit

from langchain_community.tools.gmail.create_draft import GmailCreateDraft
from langchain_community.tools.gmail.get_message import GmailGetMessage
from langchain_community.tools.gmail.get_thread import GmailGetThread
from langchain_community.tools.gmail.search import GmailSearch
from langchain_community.tools.gmail.send_message import GmailSendMessage
from langchain_community.tools.gmail.utils import build_resource_service

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
    """Toolkit for interacting with Gmail.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by reading, creating, updating, deleting
        data associated with this service.

        For example, this toolkit can be used to send emails on behalf of the
        associated account.

        See https://python.langchain.com/docs/security for more information.

    Setup:
        You will need a Google credentials.json file to use this toolkit.
        See instructions here: https://python.langchain.com/v0.2/docs/integrations/tools/gmail/#setup

    Key init args:
        api_resource: Optional. The Google API resource. Default is None.

    Instantiate:
        .. code-block:: python

            from langchain_google_community import GmailToolkit

            toolkit = GmailToolkit()

    Tools:
        .. code-block:: python

            toolkit.get_tools()

        .. code-block:: none

            [GmailCreateDraft(api_resource=<googleapiclient.discovery.Resource object at 0x1094509d0>),
            GmailSendMessage(api_resource=<googleapiclient.discovery.Resource object at 0x1094509d0>),
            GmailSearch(api_resource=<googleapiclient.discovery.Resource object at 0x1094509d0>),
            GmailGetMessage(api_resource=<googleapiclient.discovery.Resource object at 0x1094509d0>),
            GmailGetThread(api_resource=<googleapiclient.discovery.Resource object at 0x1094509d0>)]

    Use within an agent:
        .. code-block:: python

            from langchain_openai import ChatOpenAI
            from langgraph.prebuilt import create_react_agent

            llm = ChatOpenAI(model="gpt-4o-mini")

            agent_executor = create_react_agent(llm, tools)

            example_query = "Draft an email to fake@fake.com thanking them for coffee."

            events = agent_executor.stream(
                {"messages": [("user", example_query)]},
                stream_mode="values",
            )
            for event in events:
                event["messages"][-1].pretty_print()

        .. code-block:: none

             ================================[1m Human Message [0m=================================

            Draft an email to fake@fake.com thanking them for coffee.
            ==================================[1m Ai Message [0m==================================
            Tool Calls:
            create_gmail_draft (call_slGkYKZKA6h3Mf1CraUBzs6M)
            Call ID: call_slGkYKZKA6h3Mf1CraUBzs6M
            Args:
                message: Dear Fake,

            I wanted to take a moment to thank you for the coffee yesterday. It was a pleasure catching up with you. Let's do it again soon!

            Best regards,
            [Your Name]
                to: ['fake@fake.com']
                subject: Thank You for the Coffee
            =================================[1m Tool Message [0m=================================
            Name: create_gmail_draft

            Draft created. Draft Id: r-7233782721440261513
            ==================================[1m Ai Message [0m==================================

            I have drafted an email to fake@fake.com thanking them for the coffee. You can review and send it from your email draft with the subject "Thank You for the Coffee".

    Parameters:
        api_resource: Optional. The Google API resource. Default is None.
    """  # noqa: E501

    api_resource: Resource = Field(default_factory=build_resource_service)

    class Config:
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
