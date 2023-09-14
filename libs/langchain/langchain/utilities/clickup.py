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

        # Get all the teams that the user has access to
        url = "https://api.clickup.com/api/v2/team"

        headers = {"Authorization": values["access_token"]}

        response = requests.get(url, headers=headers)

        data = response.json()
        if "teams" in data.keys() and len(data["teams"]) > 0:
            values["team_id"] = data["teams"][0]["id"]

        return values


    def parse_task(self, data):
        """
            Formats a task
        """
        parsed = []
        for issue in issues["issues"]:
            key = issue["key"]
            summary = issue["fields"]["summary"]
            created = issue["fields"]["created"][0:10]
            priority = issue["fields"]["priority"]["name"]
            status = issue["fields"]["status"]["name"]
            try:
                assignee = issue["fields"]["assignee"]["displayName"]
            except Exception:
                assignee = "None"
            rel_issues = {}
            for related_issue in issue["fields"]["issuelinks"]:
                if "inwardIssue" in related_issue.keys():
                    rel_type = related_issue["type"]["inward"]
                    rel_key = related_issue["inwardIssue"]["key"]
                    rel_summary = related_issue["inwardIssue"]["fields"]["summary"]
                if "outwardIssue" in related_issue.keys():
                    rel_type = related_issue["type"]["outward"]
                    rel_key = related_issue["outwardIssue"]["key"]
                    rel_summary = related_issue["outwardIssue"]["fields"]["summary"]
                rel_issues = {"type": rel_type, "key": rel_key, "summary": rel_summary}
            parsed.append(
                {
                    "key": key,
                    "summary": summary,
                    "created": created,
                    "assignee": assignee,
                    "priority": priority,
                    "status": status,
                    "related_issues": rel_issues,
                }
            )
        return parsed

    def parse_teams(self, data):
        """
            Parse appropriate content from the list of teams
        """
        pass 

    def parse_folders(self, data):
        """
            Parse appropriate content from the list of folders
        """
        pass

    def parse_spaces(self, data):
        """
            Parse appropriate content from the list of spaces
        """
        pass

    


    def get_authorized_teams(self) -> str:
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
    
    def update_task(self, query: str) -> str:
        """
            Update an attribute of a specified task
        """
        task = self.get_task(query)
        
        params = json.loads(query)
        url = "https://api.clickup.com/api/v2/task/" + params['task_id']

        query = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true"
        }

        headers = {"Content-Type": "application/json", "Authorization": self.access_token}
        payload = {params['attribute_name']: params['new_value']}
        
        response = requests.put(url, headers=headers, params=query, json=payload)

        print(response)
        return response

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
        elif mode == "get_spaces":
            return self.get_spaces(query)
        elif mode == "update_task":
            return self.update_task(query)
        else:
            raise ValueError(f"Got unexpected mode {mode}")

