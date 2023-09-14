"""Util that calls clickup."""
from typing import Any, Dict, List, Optional

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env
import requests
import json

class ClickupAPIWrapper(BaseModel):
    """Wrapper for Clickup API."""

    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    redirect_url: Optional[str] = None
    code: Optional[str] = None
    access_token: Optional[str] = None
    url: Optional[str] = "https://api.clickup.com/api/v2/oauth/token"
    team_id: Optional[str] = None
 
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
    
    def post_init(self) -> None:
        self.team_id = "9013051928"

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        oauth_client_secret = get_from_dict_or_env(values, "oauth_client_secret", "ouath_client_secret")
        oauth_client_id = get_from_dict_or_env(
            values, "oauth_client_id", "oauth_client_id"
        )
        code = get_from_dict_or_env(values, "code", "code")

        # url = "https://api.clickup.com/api/v2/oauth/token" # TODO: can we define this as a default and allow passing in?

        #TODO: You could ask for the code, client_id and secret and use those values to generate the access token or you could ask the user to provide that upfront
        # query = {
        #     "client_id": oauth_client_id,
        #     "client_secret": oauth_client_secret,
        #     "code": code,
        # }

        # response = requests.post(url, params=query)
        # data = response.json()
        # print(data)

        values["oauth_client_secret"] = oauth_client_secret
        values["oauth_client_id"] = oauth_client_id
        values["code"] = code
        values["access_token"] = "61681706_dc747044a6941fc9aa645a4f3bca2ba5576e7dfb516a3d1889553fe96a4084f6"
        values["team_id"] = "9013051928"
        return values


    def process_task(self, data):
        """
            Formats a task
        """
        pass


    def get_authorized_teams(self, query: str) -> str:
        """
            Get all teams for the user
        """
        url = "https://api.clickup.com/api/v2/team"

        headers = {"Authorization": self.access_token}

        response = requests.get(url, headers=headers)

        data = response.json()
        return data


    def get_spaces(self, query: str) -> str:
        """
            Get all spaces for the team 
        """
        url = "https://api.clickup.com/api/v2/team/" + self.team_id + "/space"

        query = {
            "archived": "false"
        }

        headers = {"Authorization": self.access_token}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        return data


    def get_folders(self, query: str) -> str:
        """
            Get all the folders for the team
        """
        url = "https://api.clickup.com/api/v2/team/" + self.team_id + "/space"

        query = {
        "archived": "false"
        }

        headers = {"Authorization": self.access_token}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        return data


    def get_task(self, query: str) -> str:
        """
            Retrieve a specific task 
        """

        params = json.loads(query)
        url = "https://api.clickup.com/api/v2/task/" + params['task_id']

        query = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true"
        }

        headers = {"Authorization": self.access_token}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        return data


    def query_tasks(self, query: str) -> str:
        """
            Query tasks that match certain fields
        """
        params = json.loads(query)
        url = "https://api.clickup.com/api/v2/list/" + params['list_id'] + "/task"

        query = {}

        headers = {"Authorization": self.access_token}

        response = requests.get(url, headers=headers, params=query)

        data = response.json()
        return data

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

