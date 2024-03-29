from typing import List

from langchain_core.pydantic_v1 import root_validator
from langchain_core.tools import BaseTool

from langchain_community.agent_toolkits.base import BaseToolkit
from langchain_community.tools.connery import ConneryService


class ConneryToolkit(BaseToolkit):
    """
    A LangChain Toolkit with a list of Connery Actions as tools.
    """

    tools: List[BaseTool]

    def get_tools(self) -> List[BaseTool]:
        """
        Returns the list of Connery Actions.
        """
        return self.tools

    @root_validator()
    def validate_attributes(cls, values: dict) -> dict:
        """
        Validate the attributes of the ConneryToolkit class.
        Parameters:
            values (dict): The arguments to validate.
        Returns:
            dict: The validated arguments.
        """

        if not values.get("tools"):
            raise ValueError("The attribute 'tools' must be set.")

        return values

    @classmethod
    def create_instance(cls, connery_service: ConneryService) -> "ConneryToolkit":
        """
        Creates a Connery Toolkit using a Connery Service.
        Parameters:
            connery_service (ConneryService): The Connery Service
            to to get the list of Connery Actions.
        Returns:
            ConneryToolkit: The Connery Toolkit.
        """

        instance = cls(tools=connery_service.list_actions())

        return instance
