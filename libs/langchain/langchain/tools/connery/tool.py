"""
This module contains the Connery Action Tool.
"""

import json
import requests
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.tools import BaseTool
from langchain.utils.env import get_from_dict_or_env
from typing import Dict, List, Type, Optional
from .models import Action, Parameter, Input, Output

class ConneryAction(BaseTool):
    """
    A LangChain Tool wrapping a Connery Action.
    """

    @root_validator()
    def validate_attributes(cls, values: dict) -> dict:
        """
        Validate the attributes of the ConneryAction class.
        Parameters:
            values (dict): The arguments to validate.
        Returns:
            dict: The validated arguments.
        """

        if not values.get('name'):
            raise ValueError("The name of the action must be set.")
        if not values.get('description'):
            raise ValueError("The description of the action must be set.")
        if not values.get('args_schema'):
            raise ValueError("The args_schema of the action must be set.")
        if not values.get('_run'):
            raise ValueError("The _run function of the action must be set.")
        
        return values


    @classmethod
    def init(cls, action: Action, connery_service: "ConneryService"):
        """
        Initialize a Connery Action Tool.
        Parameters:
            action (Action): The Connery Action to wrap.
            connery_service (ConneryService): The Connery API Wrapper.
        Returns:
            ConneryAction: The Connery Action Tool.
        """

        input_schema = cls.create_dynamic_input_schema(action.inputParameters)
        description = action.title + (": " + action.description if action.description else "")

        return cls(
            name=action.id,
            description=description,
            args_schema=input_schema,
            _run=lambda input: connery_service.run_action(action.id, None, input)
        )
    
    @classmethod
    def create_dynamic_input_schema(cls, inputParameters: List[Parameter]) -> Type[BaseModel]:
        """
        Creates a dynamic input schema for a Connery Action Tool based on the input parameters of the Connery Action.
        Parameters:
            inputParameters: List of input parameters of the Connery Action.
        Returns:
            Type[BaseModel]: The dynamic input schema for the Connery Action Tool.
        """

        dynamic_input_fields = {}

        for param in inputParameters:
            field_info = {}
            
            if param.description:
                field_info['description'] = param.description
            else:
                field_info['description'] = param.title
            
            if param.validation and param.validation.required:
                field_info['default'] = ...
            else:
                field_info['default'] = None
            
            dynamic_input_fields[param.key] = (str, Field(**field_info))

        InputModel = create_model('InputModel', **dynamic_input_fields)
        DynamicInputModel = create_model('DynamicInputModel', input=(InputModel, Field(..., description="The list of input parameters expected by the action to run.")))

        return DynamicInputModel
    
class ConneryService(BaseModel):
    """
    A service for working with Connery actions.
    """

    runner_url: Optional[str] = None
    api_key: Optional[str] = None

    @root_validator()
    def validate_attributes(cls, values: Dict) -> Dict:
        """
        Validate the attributes of the ConneryService class.
        Parameters:
            values (dict): The arguments to validate.
        Returns:
            dict: The validated arguments.
        """

        runner_url = get_from_dict_or_env(values, "runner_url", "CONNERY_RUNNER_URL")
        api_key = get_from_dict_or_env(values, "api_key", "CONNERY_RUNNER_API_KEY")
        
        if not runner_url:
            raise ValueError("CONNERY_RUNNER_URL environment variable must be set.")
        if not api_key:
            raise ValueError("CONNERY_RUNNER_API_KEY environment variable must be set.")

        values['runner_url'] = runner_url
        values['api_key'] = api_key

        return values
    
    def list_actions(self) -> List[ConneryAction]:
        """
        Returns the list of actions available in the Connery Runner.
        Returns:
            List[ConneryAction]: The list of actions available in the Connery Runner.
        """

        return [ConneryAction.init(action, self) for action in self._list_actions()]

    def get_action(self, action_id: str) -> ConneryAction:
        """
        Returns the specified action available in the Connery Runner.
        Parameters:
            action_id (str): The ID of the action to return.
        Returns:
            ConneryAction: The action with the specified ID.
        """

        return ConneryAction.init(self._get_action(action_id), self)

    def run_action(self, action_id: str, input: Input = None) -> Output:
        """
        Runs the specified Connery Action with the provided input.
        Parameters:
            action_id (str): The ID of the action to run.
            input (Input): The input object expected by the action.
        Returns:
            Output: The output of the action.
        """

        return self._run_action(action_id, None, input)

    def _list_actions(self) -> List[Action]:
        """
        Returns the list of actions available in the Connery Runner.
        Returns:
            List[Action]: The list of actions available in the Connery Runner.
        """

        response = requests.get(
            f"{self.runner_url}/v1/actions",
            headers=self._get_headers()
        )

        if not response.ok:
            raise ValueError(f"Failed to list actions. Status code: {response.status_code}. Error message: {response.json()['error']['message']}")

        return [Action(**action) for action in response.json()['data']]

    def _get_action(self, action_id: str) -> Action:
        """
        Returns the specified action available in the Connery Runner.
        Parameters:
            action_id (str): The ID of the action to return.
        Returns:
            Action: The action with the specified ID.
        """

        actions = self._list_actions()
        action = next((action for action in actions if action.id == action_id), None)
        if not action:
            raise ValueError(f"The action with ID {action_id} was not found in the list of available actions in the Connery Runner.")
        return action

    def _run_action(self, action_id: str, prompt: str = None, input: Input = None) -> Output:
        """
        Runs the specified Connery Action with the provided input.
        Parameters:
            action_id (str): The ID of the action to run.
            prompt (str): This is a plain English prompt with all the information needed to run the action.
            input (Input): The input object expected by the action. If provided together with the prompt, the input takes precedence over the input specified in the prompt.
        Returns:
            Output: The output of the action.
        """

        response = requests.post(
            f"{self.runner_url}/v1/actions/{action_id}/run",
            headers=self._get_headers(),
            data=json.dumps({"prompt": prompt, "input": input})
        )

        if not response.ok:
            raise ValueError(f"Failed to run action. Status code: {response.status_code}. Error message: {response.json()['error']['message']}")

        if not response.json()['data']['output']:
            return Output(**{"__root__": {}})
        else:
            return Output(**response.json()['data']['output'])

    def _get_headers(self) -> Dict[str, str]:
        """
        Returns a standard set of HTTP headers to be used in API calls to the Connery runner.
        Returns:
            Dict[str, str]: The standard set of HTTP headers.
        """

        return {
            "Content-Type": "application/json",
            "x-api-key": self.api_key
        }
