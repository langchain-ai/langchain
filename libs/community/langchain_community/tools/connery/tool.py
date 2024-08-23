import asyncio
from functools import partial
from typing import Any, Dict, List, Optional, Type

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.tools import BaseTool

from langchain_community.tools.connery.models import Action, Parameter


class ConneryAction(BaseTool):
    """Connery Action tool."""

    name: str
    description: str
    args_schema: Type[BaseModel]

    action: Action
    connery_service: Any

    def _run(
        self,
        run_manager: Optional[CallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Runs the Connery Action with the provided input.
        Parameters:
            kwargs (Dict[str, str]): The input dictionary expected by the action.
        Returns:
            Dict[str, str]: The output of the action.
        """

        return self.connery_service.run_action(self.action.id, kwargs)

    async def _arun(
        self,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
        **kwargs: Any,
    ) -> Dict[str, str]:
        """
        Runs the Connery Action asynchronously with the provided input.
        Parameters:
            kwargs (Dict[str, str]): The input dictionary expected by the action.
        Returns:
            Dict[str, str]: The output of the action.
        """

        func = partial(self._run, **kwargs)
        return await asyncio.get_event_loop().run_in_executor(None, func)

    def get_schema_json(self) -> str:
        """
        Returns the JSON representation of the Connery Action Tool schema.
        This is useful for debugging.
        Returns:
            str: The JSON representation of the Connery Action Tool schema.
        """

        return self.args_schema.schema_json(indent=2)

    @root_validator(pre=True)
    def validate_attributes(cls, values: dict) -> dict:
        """
        Validate the attributes of the ConneryAction class.
        Parameters:
            values (dict): The arguments to validate.
        Returns:
            dict: The validated arguments.
        """

        # Import ConneryService here and check if it is an instance
        # of ConneryService to avoid circular imports
        from .service import ConneryService

        if not isinstance(values.get("connery_service"), ConneryService):
            raise ValueError(
                "The attribute 'connery_service' must be an instance of ConneryService."
            )

        if not values.get("name"):
            raise ValueError("The attribute 'name' must be set.")
        if not values.get("description"):
            raise ValueError("The attribute 'description' must be set.")
        if not values.get("args_schema"):
            raise ValueError("The attribute 'args_schema' must be set.")
        if not values.get("action"):
            raise ValueError("The attribute 'action' must be set.")
        if not values.get("connery_service"):
            raise ValueError("The attribute 'connery_service' must be set.")

        return values

    @classmethod
    def create_instance(cls, action: Action, connery_service: Any) -> "ConneryAction":
        """
        Creates a Connery Action Tool from a Connery Action.
        Parameters:
            action (Action): The Connery Action to wrap in a Connery Action Tool.
            connery_service (ConneryService): The Connery Service
            to run the Connery Action. We use Any here to avoid circular imports.
        Returns:
            ConneryAction: The Connery Action Tool.
        """

        # Import ConneryService here and check if it is an instance
        # of ConneryService to avoid circular imports
        from .service import ConneryService

        if not isinstance(connery_service, ConneryService):
            raise ValueError(
                "The connery_service must be an instance of ConneryService."
            )

        input_schema = cls._create_input_schema(action.inputParameters)
        description = action.title + (
            ": " + action.description if action.description else ""
        )

        instance = cls(
            name=action.id,
            description=description,
            args_schema=input_schema,
            action=action,
            connery_service=connery_service,
        )

        return instance

    @classmethod
    def _create_input_schema(cls, inputParameters: List[Parameter]) -> Type[BaseModel]:
        """
        Creates an input schema for a Connery Action Tool
        based on the input parameters of the Connery Action.
        Parameters:
            inputParameters: List of input parameters of the Connery Action.
        Returns:
            Type[BaseModel]: The input schema for the Connery Action Tool.
        """

        dynamic_input_fields: Dict[str, Any] = {}

        for param in inputParameters:
            default = ... if param.validation and param.validation.required else None
            title = param.title
            description = param.title + (
                ": " + param.description if param.description else ""
            )
            type = param.type

            dynamic_input_fields[param.key] = (
                str,
                Field(default, title=title, description=description, type=type),
            )

        InputModel = create_model("InputSchema", **dynamic_input_fields)
        return InputModel
