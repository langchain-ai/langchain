from typing import Any, List

from langchain_core.tools import BaseTool
from langchain_core.tools.base import BaseToolkit
from pydantic import model_validator

from langchain_community.tools.connery import ConneryService


class ConneryToolkit(BaseToolkit):
    """
    Toolkit with a list of Connery Actions as tools.

    Parameters:
        tools (List[BaseTool]): The list of Connery Actions.
    """

    tools: List[BaseTool]

    def get_tools(self) -> List[BaseTool]:
        """
        Returns the list of Connery Actions.
        """
        return self.tools

    @model_validator(mode="before")
    @classmethod
    def validate_attributes(cls, values: dict) -> Any:
        """
        Validate the attributes of the ConneryToolkit class.

        Args:
            values (dict): The arguments to validate.
        Returns:
            dict: The validated arguments.

        Raises:
            ValueError: If the 'tools' attribute is not set
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
                to get the list of Connery Actions.
        Returns:
            ConneryToolkit: The Connery Toolkit.
        """

        instance = cls(tools=connery_service.list_actions())  # type: ignore[arg-type]

        return instance
