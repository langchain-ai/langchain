import json
import requests
from typing import Optional, List, Dict
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain.utils.env import get_from_dict_or_env

class Validation(BaseModel):
    required: Optional[bool] = None

class Parameter(BaseModel):
    key: str
    title: str
    description: Optional[str] = None
    type: str
    validation: Optional[Validation] = None

class Action(BaseModel):
    id: str
    key: str
    title: str
    description: Optional[str] = None
    type: str
    inputParameters: List[Parameter]
    outputParameters: List[Parameter]
    pluginId: str

class Input(BaseModel):
    __root__: Dict[str, str]

class Output(BaseModel):
    __root__: Dict[str, str]

class ConneryAPIWrapper(BaseModel):
    """A wrapper for the Connery API."""

    runner_url: Optional[str] = None
    api_key: Optional[str] = None

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate the environment variables and set the values if they are not set."""

        runner_url = get_from_dict_or_env(values, "runner_url", "CONNERY_RUNNER_URL")
        api_key = get_from_dict_or_env(values, "api_key", "CONNERY_RUNNER_API_KEY")

        if not runner_url or not api_key:
            raise ValueError("CONNERY_RUNNER_URL and CONNERY_RUNNER_API_KEY environment variables must be set.")

        values['runner_url'] = runner_url
        values['api_key'] = api_key

        return values
    

    def list_actions(self) -> List[Action]:
        """
        Returns the list of actions available in the Connery runner.
        Returns:
            List[Action]: The list of actions available in the Connery runner.
        """

        response = requests.get(
            f"{self.runner_url}/v1/actions",
            headers=self._get_headers()
        )

        if not response.ok:
            raise ValueError(f"Failed to list actions. Status code: {response.status_code}. Error message: {response.json()['error']['message']}")

        return [Action(**action) for action in response.json()['data']]

    def get_action(self, action_id: str) -> Action:
        """
        Returns the specified action available in the Connery runner.
        Parameters:
            action_id (str): The ID of the action to return.
        Returns:
            Action: The action with the specified ID.
        """

        actions = self.list_actions()
        action = next((action for action in actions if action.id == action_id), None)
        if not action:
            raise ValueError(f"The action with ID {action_id} was not found in the list of available actions in the Connery runner.")
        return action

    def run_action(self, action_id: str, prompt: str = None, input: Input = None) -> Output:
        """
        Runs the specified Connery action with the provided input.
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
