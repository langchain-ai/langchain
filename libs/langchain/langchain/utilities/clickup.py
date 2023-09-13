"""Util that calls clickup."""
from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env


class ClickupAPIWrapper(BaseModel):
    """Wrapper for Clickup API."""

    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    redirect_url: Optional[str] = None
    code: Optional[str] = None
    access_token: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        oauth_client_secret = get_from_dict_or_env(values, "oauth_client_secret", "client_secret")
        values["oauth_client_secret"] = oauth_client_secret

        oauth_client_id = get_from_dict_or_env(
            values, "oauth_client_id", "oauth_client_id"
        )
        values["oauth_client_id"] = oauth_client_id

        code = get_from_dict_or_env(values, "code", "code")
        values["code"] = code

        try:
            import requests
        except ImportError:
            raise ImportError(
                "requests is not installed. "
                "Please install it with `pip install requests`"
            )

        #TODO: You could ask for the code, client_id and secret and use those values to generate the access token or you could ask the user to provide that upfront

        url = "https://api.clickup.com/api/v2/oauth/token"

        query = {
            "client_id": oauth_client_id,
            "client_secret": oauth_client_secret,
            "code": code,
        }

        response = requests.post(url, params=query)
        data = response.json()

        values["access_token"] = data["access_token"]
        return values


    def get_authorized_teams(self, query: str) -> str:
        """
            Get all teams for the user
        """
        url = "https://api.clickup.com/api/v2/team"

        headers = {"Authorization": values["access_token"]}

        response = requests.get(url, headers=headers)

        data = response.json()
        print(data)


    def get_spaces(self, query: str) -> str:
        """
            Get all spaces for the team 
        """
        url = "https://api.clickup.com/api/v2/team/" + team_id + "/space"

        query = {
            "archived": "false"
        }

        headers = {"Authorization": values["access_token"]}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        print(data)


    def get_folders(self, query: str) -> str:
        """
            Get all the folders for the team
        """
        url = "https://api.clickup.com/api/v2/team/" + team_id + "/space"

        query = {
        "archived": "false"
        }

        headers = {"Authorization": values["access_token"]}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        print(data)


    def get_task(self, query: str) -> str:
        """
            Retrieve a specific task 
        """
        url = "https://api.clickup.com/api/v2/task/" + task_id

        query = {
            "custom_task_ids": "true",
            "team_id": "9013051928",
            "include_subtasks": "true"
        }

        headers = {"Authorization": values["access_token"]}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        print(data)


    def query_tasks(self, query: str) -> str:
        """
            Query tasks that match certain fields
        """
        url = "https://api.clickup.com/api/v2/list/" + list_id + "/task"

        query = {}

        headers = {"Authorization": values["access_token"]}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        print(data)


    def run(self, mode: str, query: str) -> str:
        if mode == "get_task":
            return self.get_task(query)
        elif mode == "get_teams":
            return self.get_authorized_teams()
        elif mode == "create_task":
            return self.create_task(query)
        elif mode == "get_list":
            return self.get_list(query)
        elif mode == "get_folders":
            return self.get_folders(query)
        else:
            raise ValueError(f"Got unexpected mode {mode}")

