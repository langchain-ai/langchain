"""Util that calls Jira."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.tools.jira.prompt import (
    JIRA_CATCH_ALL_PROMPT,
    JIRA_GET_ALL_PROJECTS_PROMPT,
    JIRA_ISSUE_CREATE_PROMPT,
    JIRA_JQL_PROMPT,
)
from langchain.utils import get_from_dict_or_env


# TODO: think about error handling, more specific api specs, and jql/project limits
class JiraAPIWrapper(BaseModel):
    """Wrapper for Jira API."""

    jira: Any  #: :meta private:
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    jira_instance_url: Optional[str] = None

    operations: List[Dict] = [
        {
            "mode": "jql",
            "name": "JQL Query",
            "description": JIRA_JQL_PROMPT,
        },
        {
            "mode": "get_projects",
            "name": "Get Projects",
            "description": JIRA_GET_ALL_PROJECTS_PROMPT,
        },
        {
            "mode": "create_issue",
            "name": "Create Issue",
            "description": JIRA_ISSUE_CREATE_PROMPT,
        },
        {
            "mode": "other",
            "name": "Catch all Jira API call",
            "description": JIRA_CATCH_ALL_PROMPT,
        },
    ]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def list(self) -> List[Dict]:
        return self.operations

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        jira_username = get_from_dict_or_env(values, "jira_username", "JIRA_USERNAME")
        values["jira_username"] = jira_username

        jira_api_token = get_from_dict_or_env(
            values, "jira_api_token", "JIRA_API_TOKEN"
        )
        values["jira_api_token"] = jira_api_token

        jira_instance_url = get_from_dict_or_env(
            values, "jira_instance_url", "JIRA_INSTANCE_URL"
        )
        values["jira_instance_url"] = jira_instance_url

        try:
            from atlassian import Jira
        except ImportError:
            raise ImportError(
                "atlassian-python-api is not installed. "
                "Please install it with `pip install atlassian-python-api`"
            )

        jira = Jira(
            url=jira_instance_url,
            username=jira_username,
            password=jira_api_token,
            cloud=True,
        )
        values["jira"] = jira

        return values

    def parse_issues(self, issues: Dict) -> List[dict]:
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

    def parse_projects(self, projects: List[dict]) -> List[dict]:
        parsed = []
        for project in projects:
            id = project["id"]
            key = project["key"]
            name = project["name"]
            type = project["projectTypeKey"]
            style = project["style"]
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

    def create(self, query: str) -> str:
        try:
            import json
        except ImportError:
            raise ImportError(
                "json is not installed. Please install it with `pip install json`"
            )
        params = json.loads(query)
        return self.jira.issue_create(fields=dict(params))

    def other(self, query: str) -> str:
        context = {"self": self}
        exec(f"result = {query}", context)
        result = context["result"]
        return str(result)

    def run(self, mode: str, query: str) -> str:
        if mode == "jql":
            return self.search(query)
        elif mode == "get_projects":
            return self.project()
        elif mode == "create_issue":
            return self.create(query)
        elif mode == "other":
            return self.other(query)
        else:
            raise ValueError(f"Got unexpected mode {mode}")
