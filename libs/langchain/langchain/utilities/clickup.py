"""Util that calls clickup."""
from typing import Any, Dict, Optional, Tuple

from langchain.pydantic_v1 import BaseModel, Extra, root_validator
from langchain.utils import get_from_dict_or_env
import requests
import json
import warnings

from dataclasses import dataclass, fields, asdict
from typing import List, Dict, Any, Optional, Union


@dataclass
class Task:
    id: int
    name: str
    text_content: str
    description: str
    status: str
    creator_id: int
    creator_username: str
    creator_email: str
    assignees: List[Dict[str, Any]]
    watcher_username: str
    watcher_email: str
    priority: str
    due_date: Optional[str]
    start_date: Optional[str]
    points: int
    team_id: int
    project_id: int

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> 'Task':
        return cls(
            id=data['id'],
            name=data['name'],
            text_content=data['text_content'],
            description=data['description'],
            status=data['status']['status'],
            creator_id=data['creator']['id'],
            creator_username=data['creator']['username'],
            creator_email=data['creator']['email'],
            assignees=data['assignees'],
            watcher_username=data['watchers'][0]['username'],
            watcher_email=data['watchers'][0]['email'],
            priority=data['priority']['priority'],
            due_date=data['due_date'],
            start_date=data['start_date'],
            points=data['points'],
            team_id=data['team_id'],
            project_id=data['project']['id']
        )


@dataclass
class CUList:
    folder_id: float
    name: str
    content: Optional[str] = None
    due_date: Optional[int] = None
    due_date_time: Optional[bool] = None
    priority: Optional[int] = None
    assignee: Optional[int] = None
    status: Optional[str] = None

    @classmethod
    def from_data(cls, data: dict) -> 'CUList':
        return cls(
            folder_id=data['folder_id'],
            name=data['name'],
            content=data.get('content'),
            due_date=data.get('due_date'),
            due_date_time=data.get('due_date_time'),
            priority=data.get('priority'),
            assignee=data.get('assignee'),
            status=data.get('status')
        )
        
@dataclass
class Member:
    id: int
    username: str
    email: str
    initials: str

    @classmethod
    def from_data(cls, data: Dict) -> 'Member':
        return cls(
            id=data['user']['id'],
            username=data['user']['username'],
            email=data['user']['email'],
            initials=data['user']['initials']
        )

@dataclass
class Team:
    id: int
    name: str
    members: List[Member]

    @classmethod
    def from_data(cls, data: Dict) -> 'Team':
        members = [Member.from_data(member_data) for member_data in data['members']]
        return cls(
            id=data['id'],
            name=data['name'],
            members=members
        )

@dataclass
class Space:
    id: int
    name: str
    private: bool
    enabled_features: Dict[str, Any]

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> 'Space':
        space_data = data['spaces'][0]
        enabled_features = {
            feature: value
            for feature, value in space_data['features'].items()
            if value['enabled']
        }
        return cls(
            id=space_data['id'],
            name=space_data['name'],
            private=space_data['private'],
            enabled_features=enabled_features
        )

def parse_dict_through_dataclass(data: dict, dataclass: dataclass, fault_tolerant=False) -> dict:
    """ This is a helper function that helps us parse a dictionary by creating a dataclass and then turning it
    back into a dictionary. This might seem silly but it's a nice way to:
    1. Extract and format data from a dictionary according to a schema
    2. Provide a central place to do this in a fault tolerant way

    """
    try:
        return asdict(dataclass.from_data(data))
    except Exception as e:
        if fault_tolerant:
            warnings.warn(f'Error encountered while trying to parse {dataclass}: {e}\n Falling back to returning input data.')
            return data
        else: raise e

def extract_dict_elements_from_dataclass_fields(data: dict, dataclass: dataclass):
    output = {}
    for attribute in fields(dataclass):
            if attribute.name in data.keys():
                output[attribute.name] = data[attribute.name]
    return output


def load_params(query: str, fault_tolerant=False) -> Tuple[Optional[Any], Optional[str]]:
        """
        Attempts to parse a JSON string and return the parsed object.
        If parsing fails, returns an error message.

        :param query: The JSON string to parse.
        :return: A tuple containing the parsed object or None and an error message or None.
        """
        try:
            return json.loads(query), None
        except json.JSONDecodeError as e:
            if fault_tolerant:
                return None, f'Input must be a valid JSON. Got the following error: {str(e)}. Please reformat and try again.'
            else: raise e

def fetch_first_id(data: dict, key: str) -> int:
    if key in data and len(data[key]) > 0:
        if len(data[key]) > 1:
            warnings.warn(f'Found multiple {key}: {data[key]}. Defaulting to first.')
        return data[key][0]["id"]
    return None

def fetch_data(url: str, access_token: str, query: dict = None) -> dict:
    headers = {"Authorization": access_token}
    response = requests.get(url, headers=headers, params=query)
    response.raise_for_status()
    return response.json()

def fetch_team_id(access_token: str) -> int:
    url = "https://api.clickup.com/api/v2/team"
    data = fetch_data(url, access_token)
    return fetch_first_id(data, "teams")

def fetch_space_id(team_id: int, access_token: str) -> int:
    url = f"https://api.clickup.com/api/v2/team/{team_id}/space"
    data = fetch_data(url, access_token, query={"archived": "false"})
    return fetch_first_id(data, "spaces")

def fetch_folder_id(space_id: int, access_token: str) -> int:
    url = f"https://api.clickup.com/api/v2/space/{space_id}/folder"
    data = fetch_data(url, access_token, query={"archived": "false"})
    return fetch_first_id(data, "folders")

def fetch_list_id(space_id: int, folder_id: int, access_token: str) -> int:
    if folder_id:
        url = f"https://api.clickup.com/api/v2/folder/{folder_id}/list"
    else:
        url = f"https://api.clickup.com/api/v2/space/{space_id}/list"
    
    data = fetch_data(url, access_token, query={"archived": "false"})
    
    # The structure to fetch list id differs based on whether it's from a folder or folderless
    if folder_id and "id" in data.keys():
        return data["id"]
    else:
        return fetch_first_id(data, "lists")


class ClickupAPIWrapper(BaseModel):
    """Wrapper for Clickup API."""

    oauth_client_id: Optional[str] = None
    oauth_client_secret: Optional[str] = None
    redirect_url: Optional[str] = None
    access_token: Optional[str] = None
    url: Optional[str] = None
    team_id: Optional[str] = None
    space_id: Optional[str] = None
    folder_id: Optional[str] = None
    list_id: Optional[str] = None
 
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
    
    @classmethod
    def get_access_code_url(cls, oauth_client_id, redirect_uri='https://google.com'):
        return f"https://app.clickup.com/api?client_id={oauth_client_id}&redirect_uri={redirect_uri}"
    
    @classmethod
    def get_access_token(cls, oauth_client_id, oauth_client_secret, code):
        url = "https://api.clickup.com/api/v2/oauth/token"
        
        query = {
            "client_id": oauth_client_id,
            "client_secret": oauth_client_secret,
            "code": code,
        }

        response = requests.post(url, params=query)
        data = response.json()
        
        try:
            return data['access_token']
        except:
            print(f'Error: {data}')
            if 'ECODE' in data.keys() and data['ECODE'] == 'OAUTH_014':
                print('You already used this code once. Go back a step and generate a new code.')
                print(f'Our best guess for the url to get a new code is:\n{ClickupAPIWrapper.get_access_code_url(oauth_client_id)}')
            return None
        

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """
            Validate that api key and python package exists in environment
        """
        values["team_id"] = fetch_team_id(values["access_token"])
        values["space_id"] = fetch_space_id(values["team_id"], values["access_token"])
        values["folder_id"] = fetch_folder_id(values["space_id"], values["access_token"])
        values["list_id"] = fetch_list_id(values["space_id"], values["folder_id"], values["access_token"])

        return values

    def attempt_parse_teams(self, input_dict):
        """
            Parse appropriate content from the list of teams
        """

        parsed_teams = {'teams': []}
        for team in input_dict['teams']:
            try: 
                team = parse_dict_through_dataclass(team, Team, fault_tolerant=False)
                parsed_teams['teams'].append(team)
            except Exception as e:
                warnings.warn(f'Error parsing a team {e}')

        return parsed_teams

    def get_authorized_teams(self) -> str:
        """
            Get all teams for the user
        """
        url = "https://api.clickup.com/api/v2/team"

        headers = {"Authorization": self.access_token}

        response = requests.get(url, headers=headers)

        data = response.json()
        parsed_teams = self.attempt_parse_teams(data)

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

        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error
            
        url = "https://api.clickup.com/api/v2/task/" + params['task_id']
        query = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true"
        }
        headers = {"Authorization": self.access_token}
        response = requests.get(url, headers=headers, params=query)
        data = response.json()
        parsed_task = parse_dict_through_dataclass(data, Task, fault_tolerant=True)
        return parsed_task


    def get_lists(self, query: str) -> str:
        """
            Get all available lists
        """
        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error

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
        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error
        
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
        parsed_spaces = parse_dict_through_dataclass(data, Space, fault_tolerant=True)
        return parsed_spaces

    
    def get_task_attribute(self, query: str) -> str:
        """
            Update an attribute of a specified task
        """        
        task = self.get_task(query)
        params, _ = load_params(query, fault_tolerant=True)
        
        if params['attribute_name'] not in task.keys():
            return f"Error: attribute_name = {params['attribute_name']} was not found in task keys {task.keys()}. Please call again with one of the key names."
        return task[params['attribute_name']]

    def update_task(self, query: str) -> str:
        """
            Update an attribute of a specified task
        """        
        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error
        url = "https://api.clickup.com/api/v2/task/" + params['task_id']

        query = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true"
        }

        headers = {"Content-Type": "application/json", "Authorization": self.access_token}
        payload = {params['attribute_name']: params['value']}
        
        response = requests.put(url, headers=headers, params=query, json=payload)

        return response
        
    def update_task_assignees(self, query: str) -> str:
        """
            Add or remove assignees of a specified task
        """        
        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error
        for user in params['users']:
            if not isinstance(user, int):
                return 'All users must be integers, not strings! Got user {user} that does not follow this convention'
            
        url = "https://api.clickup.com/api/v2/task/" + params['task_id']

        query = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true"
        }

        headers = {"Content-Type": "application/json", "Authorization": self.access_token}
        if params['operation'] == 'add':
            assigne_payload = {"add": params['users'], "rem": []}
        elif params['operation'] == 'rem':
            assigne_payload = {"add": [], "rem": params['users']}
        else:
            raise ValueError(f"Invalid operation ({params['operation']}). Valid options ['add', 'rem'].")
        
        payload = {"assignees": assigne_payload}
        response = requests.put(url, headers=headers, params=query, json=payload)
        return response


    def create_task(self, query: str) -> str:
        """
            Creates a new task
        """ 
        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error


        list_id = self.list_id
        url = "https://api.clickup.com/api/v2/list/" + list_id + "/task"
        query = {
            "custom_task_ids": "true",
            "team_id": self.team_id
        }
            
        payload = extract_dict_elements_from_dataclass_fields(params, Task)
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token,
        }
        
        response = requests.post(url, json=payload, headers=headers, params=query)
        data = response.json()
        return parse_dict_through_dataclass(data, Task, fault_tolerant=True)

    
    def create_list(self, query:str) -> str:
        """
            Creates a new list
        """ 
        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error

        # Default to using folder as location if it exists. If not, fall back to using the space
        location = self.folder_id if self.folder_id else self.space_id
        url = "https://api.clickup.com/api/v2/folder/" + location + "/list"
        
        payload = extract_dict_elements_from_dataclass_fields(params, Task)
        headers = {
            "Content-Type": "application/json",
            "Authorization": self.access_token
        }
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        parsed_list = parse_dict_through_dataclass(data, CUList, fault_tolerant=True)
        return parsed_list


    def create_folder(self, query:str) -> str:
        """
            Creates a new folder
        """ 

        params, error = load_params(query, fault_tolerant=True)
        if params is None:
            return error

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
        elif mode == "get_task_attribute":
            return self.get_task_attribute(query)
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
        elif mode == "update_task_assignees":
            return self.update_task_assignees(query)
        else:
            raise ValueError(f"Got unexpected mode {mode}")

