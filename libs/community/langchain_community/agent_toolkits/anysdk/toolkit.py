from typing import List

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools import BaseTool
from langchain_community.tools.anysdk.tool import AnySdkAction
from langchain_community.utilities.anysdk import AnySdkWrapper


class AnySdkToolkit(BaseToolkit):
    """AnySdk Toolkit.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by reading, creating, updating, deleting
        data associated with this service.

        See https://python.langchain.com/docs/security for more information.
    """

    tools: List[BaseTool] = []

    @classmethod
    def from_anysdk_api_wrapper(
        cls, anysdk_api_wrapper: AnySdkWrapper
    ) -> "AnySdkToolkit":
        tools = [
            AnySdkAction(
                name=action["name"],
                description=action["description"] or "",
                mode=action["mode"],
                api_wrapper=anysdk_api_wrapper,
            )
            for action in anysdk_api_wrapper.operations
        ]

        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        return self.tools
