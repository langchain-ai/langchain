"""
This module contains the ConneryAction Tool.
"""

from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.tools import BaseTool
from typing import Any, Dict, List, Type
from .models import Action, Parameter

class ConneryAction(BaseTool):
    """
    A LangChain Tool wrapping a Connery Action.
    """

    name: str
    description: str
    args_schema: Type[BaseModel]

    action: Action
    connery_service: Any
    
    def _run(self, **kwargs) -> Dict[str, str]:
        """
        Runs the Connery Action with the provided input.
        Parameters:
            kwargs (Dict[str, str]): The input dictionary expected by the action.
        Returns:
            Dict[str, str]: The output of the action.
        """

        return self.connery_service.run_action(self.action.id, kwargs)

    def get_schema_json(self) -> str:
        """
        Returns the JSON representation of the Connery Action Tool schema.
        This is useful for debugging.
        Returns:
            str: The JSON representation of the Connery Action Tool schema.
        """

        return self.args_schema.schema_json(indent=2)

    @root_validator()
    def validate_attributes(cls, values: dict) -> dict:
        """
        Validate the attributes of the ConneryAction class.
        Parameters:
            values (dict): The arguments to validate.
        Returns:
            dict: The validated arguments.
        """

        # Import ConneryService here and check if it is an instance of ConneryService to avoid circular imports
        from .service import ConneryService
        if not isinstance(values.get('connery_service'), ConneryService):
            raise ValueError("The connery_service must be an instance of ConneryService.")

        if not values.get('name'):
            raise ValueError("The name attribute must be set.")
        if not values.get('description'):
            raise ValueError("The description attribute must be set.")
        if not values.get('args_schema'):
            raise ValueError("The args_schema attribute must be set.")
        if not values.get('action'):
            raise ValueError("The action attribute must be set.")
        if not values.get('connery_service'):
            raise ValueError("The connery_service attribute must be set.")
        
        return values


    @classmethod
    def init(cls, action: Action, connery_service: Any):
        """
        Initialize a Connery Action Tool.
        Parameters:
            action (Action): The Connery Action to wrap.
            connery_service (ConneryService): The Connery API Wrapper. We use Any here to avoid circular imports.
        Returns:
            ConneryAction: The Connery Action Tool.
        """

        # Import ConneryService here and check if it is an instance of ConneryService to avoid circular imports
        from .service import ConneryService
        if not isinstance(connery_service, ConneryService):
            raise ValueError("The connery_service must be an instance of ConneryService.")

        input_schema = cls._create_input_schema(action.inputParameters)
        description = action.title + (": " + action.description if action.description else "")

        instance = cls(
            name=action.id,
            description=description,
            args_schema=input_schema,
            action=action,
            connery_service=connery_service
        )

        return instance
    
    @classmethod
    def _create_input_schema(cls, inputParameters: List[Parameter]) -> Type[BaseModel]:
        """
        Creates an input schema for a Connery Action Tool based on the input parameters of the Connery Action.
        Parameters:
            inputParameters: List of input parameters of the Connery Action.
        Returns:
            Type[BaseModel]: The input schema for the Connery Action Tool.
        """

        dynamic_input_fields = {}

        for param in inputParameters:
            field_info = {}
            
            field_info['title'] = param.title
            field_info['description'] = param.description if param.description else ""
            field_info['type'] = param.type
            field_info['default'] = ... if param.validation and param.validation.required else None
            
            dynamic_input_fields[param.key] = (str, Field(**field_info))

        InputModel = create_model('InputSchema', **dynamic_input_fields)
        return InputModel
