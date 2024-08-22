"""Util that calls clickup."""

import json
import warnings
from dataclasses import asdict, dataclass, fields
from typing import Any, Dict, List, Mapping, Optional, Tuple, Type, Union

import requests
from langchain_core.pydantic_v1 import BaseModel, root_validator
from langchain_core.utils import get_from_dict_or_env

DEFAULT_URL = "https://api.clickup.com/api/v2"


@dataclass
class Component:
    """Base class for all components."""

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "Component":
        raise NotImplementedError()


@dataclass
class Task(Component):
    """Class for a task."""

    id: int
    name: str
    text_content: str
    description: str
    status: str
    creator_id: int
    creator_username: str
    creator_email: str
    assignees: List[Dict[str, Any]]
    watchers: List[Dict[str, Any]]
    priority: Optional[str]
    due_date: Optional[str]
    start_date: Optional[str]
    points: int
    team_id: int
    project_id: int

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "Task":
        priority = None if data["priority"] is None else data["priority"]["priority"]
        return cls(
            id=data["id"],
            name=data["name"],
            text_content=data["text_content"],
            description=data["description"],
            status=data["status"]["status"],
            creator_id=data["creator"]["id"],
            creator_username=data["creator"]["username"],
            creator_email=data["creator"]["email"],
            assignees=data["assignees"],
            watchers=data["watchers"],
            priority=priority,
            due_date=data["due_date"],
            start_date=data["start_date"],
            points=data["points"],
            team_id=data["team_id"],
            project_id=data["project"]["id"],
        )


@dataclass
class CUList(Component):
    """Component class for a list."""

    folder_id: float
    name: str
    content: Optional[str] = None
    due_date: Optional[int] = None
    due_date_time: Optional[bool] = None
    priority: Optional[int] = None
    assignee: Optional[int] = None
    status: Optional[str] = None

    @classmethod
    def from_data(cls, data: dict) -> "CUList":
        return cls(
            folder_id=data["folder_id"],
            name=data["name"],
            content=data.get("content"),
            due_date=data.get("due_date"),
            due_date_time=data.get("due_date_time"),
            priority=data.get("priority"),
            assignee=data.get("assignee"),
            status=data.get("status"),
        )


@dataclass
class Member(Component):
    """Component class for a member."""

    id: int
    username: str
    email: str
    initials: str

    @classmethod
    def from_data(cls, data: Dict) -> "Member":
        return cls(
            id=data["user"]["id"],
            username=data["user"]["username"],
            email=data["user"]["email"],
            initials=data["user"]["initials"],
        )


@dataclass
class Team(Component):
    """Component class for a team."""

    id: int
    name: str
    members: List[Member]

    @classmethod
    def from_data(cls, data: Dict) -> "Team":
        members = [Member.from_data(member_data) for member_data in data["members"]]
        return cls(id=data["id"], name=data["name"], members=members)


@dataclass
class Space(Component):
    """Component class for a space."""

    id: int
    name: str
    private: bool
    enabled_features: Dict[str, Any]

    @classmethod
    def from_data(cls, data: Dict[str, Any]) -> "Space":
        space_data = data["spaces"][0]
        enabled_features = {
            feature: value
            for feature, value in space_data["features"].items()
            if value["enabled"]
        }
        return cls(
            id=space_data["id"],
            name=space_data["name"],
            private=space_data["private"],
            enabled_features=enabled_features,
        )


def parse_dict_through_component(
    data: dict, component: Type[Component], fault_tolerant: bool = False
) -> Dict:
    """Parse a dictionary by creating
    a component and then turning it back into a dictionary.

    This helps with two things
    1. Extract and format data from a dictionary according to schema
    2. Provide a central place to do this in a fault-tolerant way

    """
    try:
        return asdict(component.from_data(data))
    except Exception as e:
        if fault_tolerant:
            warning_str = f"""Error encountered while trying to parse
{str(data)}: {str(e)}\n Falling back to returning input data."""
            warnings.warn(warning_str)
            return data
        else:
            raise e


def extract_dict_elements_from_component_fields(
    data: dict, component: Type[Component]
) -> dict:
    """Extract elements from a dictionary.

    Args:
        data: The dictionary to extract elements from.
        component: The component to extract elements from.

    Returns:
        A dictionary containing the elements from the input dictionary that are also
        in the component.
    """
    output = {}
    for attribute in fields(component):
        if attribute.name in data:
            output[attribute.name] = data[attribute.name]
    return output


def load_query(
    query: str, fault_tolerant: bool = False
) -> Tuple[Optional[Dict], Optional[str]]:
    """Parse a JSON string and return the parsed object.

    If parsing fails, returns an error message.

    :param query: The JSON string to parse.
    :return: A tuple containing the parsed object or None and an error message or None.

    Exceptions:
        json.JSONDecodeError: If the input is not a valid JSON string.
    """
    try:
        return json.loads(query), None
    except json.JSONDecodeError as e:
        if fault_tolerant:
            return (
                None,
                f"""Input must be a valid JSON. Got the following error: {str(e)}. 
"Please reformat and try again.""",
            )
        else:
            raise e


def fetch_first_id(data: dict, key: str) -> Optional[int]:
    """Fetch the first id from a dictionary."""
    if key in data and len(data[key]) > 0:
        if len(data[key]) > 1:
            warnings.warn(f"Found multiple {key}: {data[key]}. Defaulting to first.")
        return data[key][0]["id"]
    return None


def fetch_data(url: str, access_token: str, query: Optional[dict] = None) -> dict:
    """Fetch data from a URL."""
    headers = {"Authorization": access_token}
    response = requests.get(url, headers=headers, params=query)
    response.raise_for_status()
    return response.json()


def fetch_team_id(access_token: str) -> Optional[int]:
    """Fetch the team id."""
    url = f"{DEFAULT_URL}/team"
    data = fetch_data(url, access_token)
    return fetch_first_id(data, "teams")


def fetch_space_id(team_id: int, access_token: str) -> Optional[int]:
    """Fetch the space id."""
    url = f"{DEFAULT_URL}/team/{team_id}/space"
    data = fetch_data(url, access_token, query={"archived": "false"})
    return fetch_first_id(data, "spaces")


def fetch_folder_id(space_id: int, access_token: str) -> Optional[int]:
    """Fetch the folder id."""
    url = f"{DEFAULT_URL}/space/{space_id}/folder"
    data = fetch_data(url, access_token, query={"archived": "false"})
    return fetch_first_id(data, "folders")


def fetch_list_id(space_id: int, folder_id: int, access_token: str) -> Optional[int]:
    """Fetch the list id."""
    if folder_id:
        url = f"{DEFAULT_URL}/folder/{folder_id}/list"
    else:
        url = f"{DEFAULT_URL}/space/{space_id}/list"

    data = fetch_data(url, access_token, query={"archived": "false"})

    # The structure to fetch list id differs based if its folderless
    if folder_id and "id" in data:
        return data["id"]
    else:
        return fetch_first_id(data, "lists")


class ClickupAPIWrapper(BaseModel):
    """Wrapper for Clickup API."""

    access_token: Optional[str] = None
    team_id: Optional[str] = None
    space_id: Optional[str] = None
    folder_id: Optional[str] = None
    list_id: Optional[str] = None

    class Config:
        extra = "forbid"

    @classmethod
    def get_access_code_url(
        cls, oauth_client_id: str, redirect_uri: str = "https://google.com"
    ) -> str:
        """Get the URL to get an access code."""
        url = f"https://app.clickup.com/api?client_id={oauth_client_id}"
        return f"{url}&redirect_uri={redirect_uri}"

    @classmethod
    def get_access_token(
        cls, oauth_client_id: str, oauth_client_secret: str, code: str
    ) -> Optional[str]:
        """Get the access token."""
        url = f"{DEFAULT_URL}/oauth/token"

        params = {
            "client_id": oauth_client_id,
            "client_secret": oauth_client_secret,
            "code": code,
        }

        response = requests.post(url, params=params)
        data = response.json()

        if "access_token" not in data:
            print(f"Error: {data}")  # noqa: T201
            if "ECODE" in data and data["ECODE"] == "OAUTH_014":
                url = ClickupAPIWrapper.get_access_code_url(oauth_client_id)
                print(  # noqa: T201
                    "You already used this code once. Generate a new one.",
                    f"Our best guess for the url to get a new code is:\n{url}",
                )
            return None

        return data["access_token"]

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["access_token"] = get_from_dict_or_env(
            values, "access_token", "CLICKUP_ACCESS_TOKEN"
        )
        values["team_id"] = fetch_team_id(values["access_token"])
        values["space_id"] = fetch_space_id(values["team_id"], values["access_token"])
        values["folder_id"] = fetch_folder_id(
            values["space_id"], values["access_token"]
        )
        values["list_id"] = fetch_list_id(
            values["space_id"], values["folder_id"], values["access_token"]
        )

        return values

    def attempt_parse_teams(self, input_dict: dict) -> Dict[str, List[dict]]:
        """Parse appropriate content from the list of teams."""
        parsed_teams: Dict[str, List[dict]] = {"teams": []}
        for team in input_dict["teams"]:
            try:
                team = parse_dict_through_component(team, Team, fault_tolerant=False)
                parsed_teams["teams"].append(team)
            except Exception as e:
                warnings.warn(f"Error parsing a team {e}")

        return parsed_teams

    def get_headers(
        self,
    ) -> Mapping[str, Union[str, bytes]]:
        """Get the headers for the request."""
        if not isinstance(self.access_token, str):
            raise TypeError(f"Access Token: {self.access_token}, must be str.")

        headers = {
            "Authorization": str(self.access_token),
            "Content-Type": "application/json",
        }
        return headers

    def get_default_params(self) -> Dict:
        return {"archived": "false"}

    def get_authorized_teams(self) -> Dict[Any, Any]:
        """Get all teams for the user."""
        url = f"{DEFAULT_URL}/team"

        response = requests.get(url, headers=self.get_headers())

        data = response.json()
        parsed_teams = self.attempt_parse_teams(data)

        return parsed_teams

    def get_folders(self) -> Dict:
        """
        Get all the folders for the team.
        """
        url = f"{DEFAULT_URL}/team/" + str(self.team_id) + "/space"
        params = self.get_default_params()
        response = requests.get(url, headers=self.get_headers(), params=params)
        return {"response": response}

    def get_task(self, query: str, fault_tolerant: bool = True) -> Dict:
        """
        Retrieve a specific task.
        """

        params, error = load_query(query, fault_tolerant=True)
        if params is None:
            return {"Error": error}

        url = f"{DEFAULT_URL}/task/{params['task_id']}"
        params = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true",
        }
        response = requests.get(url, headers=self.get_headers(), params=params)
        data = response.json()
        parsed_task = parse_dict_through_component(
            data, Task, fault_tolerant=fault_tolerant
        )

        return parsed_task

    def get_lists(self) -> Dict:
        """
        Get all available lists.
        """

        url = f"{DEFAULT_URL}/folder/{self.folder_id}/list"
        params = self.get_default_params()
        response = requests.get(url, headers=self.get_headers(), params=params)
        return {"response": response}

    def query_tasks(self, query: str) -> Dict:
        """
        Query tasks that match certain fields
        """
        params, error = load_query(query, fault_tolerant=True)
        if params is None:
            return {"Error": error}

        url = f"{DEFAULT_URL}/list/{params['list_id']}/task"

        params = self.get_default_params()
        response = requests.get(url, headers=self.get_headers(), params=params)

        return {"response": response}

    def get_spaces(self) -> Dict:
        """
        Get all spaces for the team.
        """
        url = f"{DEFAULT_URL}/team/{self.team_id}/space"
        response = requests.get(
            url, headers=self.get_headers(), params=self.get_default_params()
        )
        data = response.json()
        parsed_spaces = parse_dict_through_component(data, Space, fault_tolerant=True)
        return parsed_spaces

    def get_task_attribute(self, query: str) -> Dict:
        """
        Update an attribute of a specified task.
        """

        task = self.get_task(query, fault_tolerant=True)
        params, error = load_query(query, fault_tolerant=True)
        if not isinstance(params, dict):
            return {"Error": error}

        if params["attribute_name"] not in task:
            return {
                "Error": f"""attribute_name = {params['attribute_name']} was not 
found in task keys {task.keys()}. Please call again with one of the key names."""
            }

        return {params["attribute_name"]: task[params["attribute_name"]]}

    def update_task(self, query: str) -> Dict:
        """
        Update an attribute of a specified task.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {"Error": error}

        url = f"{DEFAULT_URL}/task/{query_dict['task_id']}"
        params = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true",
        }
        headers = self.get_headers()
        payload = {query_dict["attribute_name"]: query_dict["value"]}

        response = requests.put(url, headers=headers, params=params, json=payload)

        return {"response": response}

    def update_task_assignees(self, query: str) -> Dict:
        """
        Add or remove assignees of a specified task.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {"Error": error}

        for user in query_dict["users"]:
            if not isinstance(user, int):
                return {
                    "Error": f"""All users must be integers, not strings!
"Got user {user} if type {type(user)}"""
                }

        url = f"{DEFAULT_URL}/task/{query_dict['task_id']}"

        headers = self.get_headers()

        if query_dict["operation"] == "add":
            assigne_payload = {"add": query_dict["users"], "rem": []}
        elif query_dict["operation"] == "rem":
            assigne_payload = {"add": [], "rem": query_dict["users"]}
        else:
            raise ValueError(
                f"Invalid operation ({query_dict['operation']}). ",
                "Valid options ['add', 'rem'].",
            )

        params = {
            "custom_task_ids": "true",
            "team_id": self.team_id,
            "include_subtasks": "true",
        }

        payload = {"assignees": assigne_payload}
        response = requests.put(url, headers=headers, params=params, json=payload)
        return {"response": response}

    def create_task(self, query: str) -> Dict:
        """
        Creates a new task.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {"Error": error}

        list_id = self.list_id
        url = f"{DEFAULT_URL}/list/{list_id}/task"
        params = {"custom_task_ids": "true", "team_id": self.team_id}

        payload = extract_dict_elements_from_component_fields(query_dict, Task)
        headers = self.get_headers()

        response = requests.post(url, json=payload, headers=headers, params=params)
        data: Dict = response.json()
        return parse_dict_through_component(data, Task, fault_tolerant=True)

    def create_list(self, query: str) -> Dict:
        """
        Creates a new list.
        """
        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {"Error": error}

        # Default to using folder as location if it exists.
        # If not, fall back to using the space.
        location = self.folder_id if self.folder_id else self.space_id
        url = f"{DEFAULT_URL}/folder/{location}/list"

        payload = extract_dict_elements_from_component_fields(query_dict, Task)
        headers = self.get_headers()

        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        parsed_list = parse_dict_through_component(data, CUList, fault_tolerant=True)
        # set list id to new list
        if "id" in parsed_list:
            self.list_id = parsed_list["id"]
        return parsed_list

    def create_folder(self, query: str) -> Dict:
        """
        Creates a new folder.
        """

        query_dict, error = load_query(query, fault_tolerant=True)
        if query_dict is None:
            return {"Error": error}

        space_id = self.space_id
        url = f"{DEFAULT_URL}/space/{space_id}/folder"
        payload = {
            "name": query_dict["name"],
        }

        headers = self.get_headers()

        response = requests.post(url, json=payload, headers=headers)
        data = response.json()

        if "id" in data:
            self.list_id = data["id"]
        return data

    def run(self, mode: str, query: str) -> str:
        """Run the API."""
        if mode == "get_task":
            output = self.get_task(query)
        elif mode == "get_task_attribute":
            output = self.get_task_attribute(query)
        elif mode == "get_teams":
            output = self.get_authorized_teams()
        elif mode == "create_task":
            output = self.create_task(query)
        elif mode == "create_list":
            output = self.create_list(query)
        elif mode == "create_folder":
            output = self.create_folder(query)
        elif mode == "get_lists":
            output = self.get_lists()
        elif mode == "get_folders":
            output = self.get_folders()
        elif mode == "get_spaces":
            output = self.get_spaces()
        elif mode == "update_task":
            output = self.update_task(query)
        elif mode == "update_task_assignees":
            output = self.update_task_assignees(query)
        else:
            output = {"ModeError": f"Got unexpected mode {mode}."}

        try:
            return json.dumps(output)
        except Exception:
            return str(output)
