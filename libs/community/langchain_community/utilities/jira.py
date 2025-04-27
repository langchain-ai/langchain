"""Util that calls Jira."""

from typing import Any, Dict, List, Optional, Union

from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel, ConfigDict, model_validator
from typing_extensions import TypedDict


class JiraOauth2Token(TypedDict):
    """Jira OAuth2 token."""

    access_token: str
    """Jira OAuth2 access token."""
    token_type: str
    """Jira OAuth2 token type ('bearer' or other)."""


class JiraOauth2(TypedDict):
    """Jira OAuth2."""

    client_id: str
    """Jira OAuth2 client ID."""
    token: JiraOauth2Token
    """Jira OAuth2 token."""


# TODO: think about error handling, more specific api specs, and jql/project limits
class JiraAPIWrapper(BaseModel):
    """
    Wrapper for Jira API. You can connect to Jira with either an API token or OAuth2.
    - with API token, you need to provide the JIRA_USERNAME and JIRA_API_TOKEN
        environment variables or arguments.
    ex: JIRA_USERNAME=your_username JIRA_API_TOKEN=your_api_token
    - with OAuth2, you need to provide the JIRA_OAUTH2 environment variable or
        argument as a dict having as fields "client_id" and "token" which is
        a dict containing at least "access_token" and "token_type".
    ex: JIRA_OAUTH2='{"client_id": "your_client_id", "token":
        {"access_token": "your_access_token","token_type": "bearer"}}'
    """

    jira: Any = None  #: :meta private:
    confluence: Any = None
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    """Jira API token when you choose to connect to Jira with api token."""
    jira_oauth2: Optional[Union[JiraOauth2, str]] = None
    """Jira OAuth2 token when you choose to connect to Jira with oauth2."""
    jira_instance_url: Optional[str] = None
    jira_cloud: Optional[bool] = None

    model_config = ConfigDict(
        extra="forbid",
    )

    @model_validator(mode="before")
    @classmethod
    def validate_environment(cls, values: Dict) -> Any:
        """Validate that api key and python package exists in environment."""
        jira_username = get_from_dict_or_env(
            values, "jira_username", "JIRA_USERNAME", default=""
        )
        values["jira_username"] = jira_username

        jira_api_token = get_from_dict_or_env(
            values, "jira_api_token", "JIRA_API_TOKEN", default=""
        )
        values["jira_api_token"] = jira_api_token

        jira_oauth2 = get_from_dict_or_env(
            values, "jira_oauth2", "JIRA_OAUTH2", default=""
        )
        values["jira_oauth2"] = jira_oauth2

        if jira_oauth2 and isinstance(jira_oauth2, str):
            try:
                import json

                jira_oauth2 = json.loads(jira_oauth2)
            except ImportError:
                raise ImportError(
                    "json is not installed. Please install it with `pip install json`"
                )
            except json.decoder.JSONDecodeError as e:
                raise ValueError(
                    f"The format of the JIRA_OAUTH2 string is "
                    f"not a valid dictionary: {e}"
                )

        jira_instance_url = get_from_dict_or_env(
            values, "jira_instance_url", "JIRA_INSTANCE_URL"
        )
        values["jira_instance_url"] = jira_instance_url

        if "jira_cloud" in values and values["jira_cloud"] is not None:
            values["jira_cloud"] = str(values["jira_cloud"])

        jira_cloud_str = get_from_dict_or_env(values, "jira_cloud", "JIRA_CLOUD")
        jira_cloud = jira_cloud_str.lower() == "true"
        values["jira_cloud"] = jira_cloud

        if jira_api_token and jira_oauth2:
            raise ValueError(
                "You have to provide either a jira_api_token or a jira_oauth2. "
                "Not both."
            )

        try:
            from atlassian import Confluence, Jira
        except ImportError:
            raise ImportError(
                "atlassian-python-api is not installed. "
                "Please install it with `pip install atlassian-python-api`"
            )

        if jira_api_token:
            if jira_username == "":
                jira = Jira(
                    url=jira_instance_url,
                    token=jira_api_token,
                    cloud=jira_cloud,
                )
            else:
                jira = Jira(
                    url=jira_instance_url,
                    username=jira_username,
                    password=jira_api_token,
                    cloud=jira_cloud,
                )

            confluence = Confluence(
                url=jira_instance_url,
                username=jira_username,
                password=jira_api_token,
                cloud=jira_cloud,
            )
        elif jira_oauth2:
            jira = Jira(
                url=jira_instance_url,
                oauth2=jira_oauth2,
                cloud=jira_cloud,
            )
            confluence = Confluence(
                url=jira_instance_url,
                oauth2=jira_oauth2,
                cloud=jira_cloud,
            )

        values["jira"] = jira
        values["confluence"] = confluence

        return values

    def parse_issues(self, issues: Dict) -> List[dict]:
        parsed = []
        for issue in issues["issues"]:
            key = issue["key"]
            summary = issue["fields"]["summary"]
            created = issue["fields"]["created"][0:10]
            if "priority" in issue["fields"]:
                priority = issue["fields"]["priority"]["name"]
            else:
                priority = None
            status = issue["fields"]["status"]["name"]
            try:
                assignee = issue["fields"]["assignee"]["displayName"]
            except Exception:
                assignee = "None"
            rel_issues = {}
            for related_issue in issue["fields"].get("issuelinks", []):
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

    def parse_projects(self, projects: List[dict]) -> List[dict]:
        parsed = []
        for project in projects:
            id = project["id"]
            key = project["key"]
            name = project["name"]
            type = project.get("projectTypeKey")
            style = project.get("style")
            parsed.append(
                {"id": id, "key": key, "name": name, "type": type, "style": style}
            )
        return parsed

    def search(self, query: str) -> str:
        issues = self.jira.jql(query)
        parsed_issues = self.parse_issues(issues)
        parsed_issues_str = (
            "Found " + str(len(parsed_issues)) + " issues:\n" + str(parsed_issues)
        )
        return parsed_issues_str

    def project(self) -> str:
        projects = self.jira.projects()
        parsed_projects = self.parse_projects(projects)
        parsed_projects_str = (
            "Found " + str(len(parsed_projects)) + " projects:\n" + str(parsed_projects)
        )
        return parsed_projects_str

    def issue_create(self, query: str) -> str:
        try:
            import json
        except ImportError:
            raise ImportError(
                "json is not installed. Please install it with `pip install json`"
            )
        params = json.loads(query)
        return self.jira.issue_create(fields=dict(params))

    def page_create(self, query: str) -> str:
        try:
            import json
        except ImportError:
            raise ImportError(
                "json is not installed. Please install it with `pip install json`"
            )
        params = json.loads(query)
        return self.confluence.create_page(**dict(params))

    def other(self, query: str) -> str:
        try:
            import json
        except ImportError:
            raise ImportError(
                "json is not installed. Please install it with `pip install json`"
            )
        params = json.loads(query)
        jira_function = getattr(self.jira, params["function"])
        return jira_function(*params.get("args", []), **params.get("kwargs", {}))

    def run(self, mode: str, query: str) -> str:
        if mode == "jql":
            return self.search(query)
        elif mode == "get_projects":
            return self.project()
        elif mode == "create_issue":
            return self.issue_create(query)
        elif mode == "other":
            return self.other(query)
        elif mode == "create_page":
            return self.page_create(query)
        else:
            raise ValueError(f"Got unexpected mode {mode}")
