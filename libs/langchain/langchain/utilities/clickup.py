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

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""

        oauth_client_secret = get_from_dict_or_env(values, "oauth_client_secret", "ouath_client_secret")
        values["oauth_client_secret"] = oauth_client_secret

        oauth_client_id = get_from_dict_or_env(
            values, "oauth_client_id", "oauth_client_id"
        )
        code = get_from_dict_or_env(values, "code", "code")

        url = "https://api.clickup.com/api/v2/oauth/token" # TODO: can we define this as a default and allow passing in?

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

        # Get the team id
        url = "https://api.clickup.com/api/v2/team"
        headers = {"Authorization": values["access_token"]}
        response = requests.get(url, headers=headers)
        data = response.json()
        if "teams" in data.keys() and len(data["teams"]) > 0:
            values["team_id"] = data["teams"][0]["id"]

        # Get the space_id 
        url = "https://api.clickup.com/api/v2/team/" + values["team_id"] + "/space"
        query = {
            "archived": "false"
        }
        headers = {"Authorization": values["access_token"]}
        response = requests.get(url, headers=headers, params=query)
        data = response.json()
        values["space_id"] = data["spaces"][0]["id"]

        # If a user has a folder, get lists in that folder
        url = "https://api.clickup.com/api/v2/space/" + values["space_id"] + "/folder"
        query = {
            "archived": "false"
        }
        headers = {"Authorization": values["access_token"]}
        response = requests.get(url, headers=headers, params=query)
        data = response.json()

        if len(data["folders"]) > 0:
            values["folder_id"] = data["id"]
            
            # Get the list_id from this folder
            url = "https://api.clickup.com/api/v2/folder/" + values["folder_id"] + "/list"
            query = {
                "archived": "false"
            }
            headers = {"Authorization": values["access_token"]}
            response = requests.get(url, headers=headers, params=query)
            data = response.json()
            values["list_id"] = data["id"]

        else:
            values["folder_id"] = ""
            # If a user doesn't have a folder, get folderless lists
            space_id = values["space_id"]
            url = "https://api.clickup.com/api/v2/space/" + space_id + "/list"
            query = {
                "archived": "false"
            }
            headers = {"Authorization": values["access_token"]}
            response = requests.get(url, headers=headers, params=query)
            data = response.json()
            values["list_id"] = data['lists'][0]["id"]

        return values


    def parse_task(self, data):
        """
            Formats a task
        """
        parsed_task = {
            'id': data['id'],
            'name': data['name'],
            'text_content': data['text_content'],
            'description': data['description'],
            'status': data['status']['status'],
            'creator_id': data['creator']['id'],
            'creator_username': data['creator']['username'],
            'creator_email': data['creator']['email'],
            'assignees': data['assignees'],
            'watcher_username': data['watchers'][0]['username'],
            'watcher_email': data['watchers'][0]['email'],
            'priority': data['priority']['priority'],
            'due_date': data['due_date'],
            'start_date': data['start_date'],
            'points': data['points'],
            'team_id': data['team_id'],
            'project_id': data['project']['id']
        }

        return parsed_task


    def parse_teams(self, input_dict):
        """
            Parse appropriate content from the list of teams
        """

        parsed_teams = {'teams': []}
        for team in input_dict['teams']:
            team_info = {
                'id': team['id'],
                'name': team['name'],
                'members': []
            }

            for member in team['members']:
                member_info = {
                    'id': member['user']['id'],
                    'username': member['user']['username'],
                    'email': member['user']['email'],
                    'initials': member['user']['initials']
                }
                team_info['members'].append(member_info)

            parsed_teams['teams'].append(team_info)

        return parsed_teams


    def parse_folders(self, data):
        """
            Parse appropriate content from the list of folders
        """
        return data


    def parse_spaces(self, data):
        """
            Parse appropriate content from the list of spaces.  
        """
        parsed_spaces = {
            'id': data['spaces'][0]['id'],
            'name': data['spaces'][0]['name'],
            'private': data['spaces'][0]['private']
        }

        # Extract features with 'enabled' equal to True
        enabled_features = {feature: value for feature, value in data['spaces'][0]['features'].items() if value['enabled']}

        # Add the enabled features to the output dictionary
        parsed_spaces['enabled_features'] = enabled_features

        return parsed_spaces

    
    def parse_lists(self, data):
        """
            Parse appropriate content from the list of lists
        """
        return data

    def get_authorized_teams(self) -> str:
        """
            Get all teams for the user
        """
        url = "https://api.clickup.com/api/v2/team"

        headers = {"Authorization": self.access_token}

        response = requests.get(url, headers=headers)

        data = response.json()
        parsed_teams = self.parse_teams(data)

        return parsed_teams


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
        print("QUERY: " + query)
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
        parsed_task = self.parse_task(data)
        return parsed_task


    def get_lists(self, query: str) -> str:

        params = json.loads(query)
        url = "https://api.clickup.com/api/v2/folder/" + self.folder_id + "/list"
        query = {
            "archived": "false"
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
        parsed_spaces = self.parse_spaces(data)
        return parsed_spaces

    
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


    def create_task(self, query: str) -> str:

        params = json.loads(query)

        list_id = self.list_id
        url = "https://api.clickup.com/api/v2/list/" + list_id + "/task"
        query = {
            "custom_task_ids": "true",
            "team_id": self.team_id
        }
        payload = {
            "name": params["name"],
            "description": params["description"],
            "status": params["status"],
            "priority": params["priority"],
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token,
        }
        response = requests.post(url, json=payload, headers=headers, params=query)
        data = response.json()
        parsed_task = self.parse_task(data)
        return parsed_task

    
    def create_list(self, query:str) -> str:

        params = json.loads(query)
        if self.folder_id:
            # Create a list in the folder
            url = "https://api.clickup.com/api/v2/folder/" + folder_id + "/list"
        else:
            # Create a list in the space
            space_id = self.space_id
            url = "https://api.clickup.com/api/v2/space/" + space_id + "/list"
        payload = {
            "name": params["name"],
            "content": params["content"],
            "priority": params["priority"],
            "status": params["status"]
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token
        }
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        parsed_list = self.parse_lists(data)
        return parsed_list


    def create_folder(self, query:str) -> str:

        params = json.loads(query)
        space_id = self.space_id
        url = "https://api.clickup.com/api/v2/space/" + space_id + "/folder"
        payload = {
            "name": params["name"],
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token
        }
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        return data
    

    def run(self, mode: str, query: str) -> str:

        if mode == "get_task":
            return self.get_task(query)
        elif mode == "get_teams":
            return self.get_authorized_teams()
        elif mode == "create_task":
            return self.create_task(query)
        elif mode == "create_list":
            return self.create_list(query)
        elif mode == "create_folder":
            return self.create_folder(query)
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

