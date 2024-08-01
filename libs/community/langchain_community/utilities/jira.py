"""Util that calls Jira."""

import json
import inspect

from typing import Any, Dict, List, Optional

from langchain_core.pydantic_v1 import BaseModel, Extra, root_validator
from langchain_core.utils import get_from_dict_or_env


# TODO: think about error handling, more specific api specs, and jql/project limits
class JiraAPIWrapper(BaseModel):
    """Wrapper for Jira API."""

    jira: Any  #: :meta private:
    confluence: Any
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    jira_instance_url: Optional[str] = None
    jira_cloud: Optional[bool] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        jira_username = get_from_dict_or_env(
            values, "jira_username", "JIRA_USERNAME", default=""
        )
        values["jira_username"] = jira_username

        jira_api_token = get_from_dict_or_env(
            values, "jira_api_token", "JIRA_API_TOKEN"
        )
        values["jira_api_token"] = jira_api_token

        jira_instance_url = get_from_dict_or_env(
            values, "jira_instance_url", "JIRA_INSTANCE_URL"
        )
        values["jira_instance_url"] = jira_instance_url

        jira_cloud = get_from_dict_or_env(values, "jira_cloud", "JIRA_CLOUD")
        values["jira_cloud"] = jira_cloud

        try:
            from atlassian import Confluence, Jira
        except ImportError:
            raise ImportError(
                "atlassian-python-api is not installed. "
                "Please install it with `pip install atlassian-python-api`"
            )

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

        values["jira"] = jira
        values["confluence"] = confluence

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

    def search(self, query: str) -> str:
        issues = self.jira.jql(query)
        parsed_issues = self.parse_issues(issues)
        parsed_issues_str = (
            "Found " + str(len(parsed_issues)) + " issues:\n" + str(parsed_issues)
        )
        return parsed_issues_str

    def group(self, query=None, exclude=None, limit=100):
        groups_dict = self.jira.get_groups(query, exclude, limit)

        clean_groups = {"header": groups_dict["header"], "total": groups_dict["total"], "groups": []}
        for group in groups_dict["groups"]:
            clean_group = {key: group[key] for key in ["name", "groupId"] if key in group}
            title = next((label["title"] for label in group.get("labels", []) if label["text"] == "Admin"), None)
            if title:
                clean_group["title"] = title
            clean_groups["groups"].append(clean_group)

        return json.dumps(clean_groups)

    def get_all_users_from_group(self, query, include_inactive_users=False, limit=100):
        if not query:
            return json.dumps({"error": "No group name provided"})

        group = json.loads(self.group(query=query, limit=1))
        if 'groups' not in group or not group['groups']:
            return json.dumps({"error": "Group does not exist"})

        start_at = min(group['total'], limit)

        users_dict = self.jira.get_all_users_from_group(query, include_inactive_users, start_at)

        cleaned_users = {
            "self": users_dict["self"],
            "maxResults": users_dict["maxResults"],
            "startAt": users_dict["startAt"],
            "total": users_dict["total"],
            "isLast": users_dict["isLast"],
            "values": [
                {
                    "displayName": user["displayName"],
                    "active": user["active"],
                    "accountType": user["accountType"],
                } for user in users_dict["values"]
            ]
        }

        return json.dumps(cleaned_users)

    def project(self):
        projects = self.jira.projects()

        refined_projects = {
            project.get('id'): {
                'key': project.get('key'),
                'name': project.get('name'),
                'description': project.get('description'),
                'projectCategory': project.get('projectCategory', {}).get('name'),
            }
            for project in projects
        }

        return json.dumps(refined_projects)

    def issue_create(self, query: str) -> str:
        params = json.loads(query)
        return self.jira.issue_create(fields=dict(params))

    def page_create(self, query: str) -> str:
        params = json.loads(query)
        return self.confluence.create_page(**dict(params))

    def other(self, query: str) -> str:
        params = json.loads(query)
        function_name = params.get("function")

        accepted_params = self.get_jira_functions(function_name)
        if isinstance(accepted_params, str) and "not found" in accepted_params:
            raise ValueError(accepted_params)

        jira_function = getattr(self.jira, function_name)

        args = params.get("args", [])
        kwargs = params.get("kwargs", {})

        presented_params = args + list(kwargs)

        if not set(presented_params).issubset(set(accepted_params)):
            raise ValueError(f"Function '{function_name}' accepts {accepted_params} parameters. "
                             f"But got: {presented_params}.")

        return jira_function(*args, **kwargs)

    def get_jira_functions(self, query=None):
        all_attributes = dir(self.jira)
        functions = [attr for attr in all_attributes if callable(getattr(self.jira, attr))
                     and not attr.startswith('_')]

        if query is None:
            return ", ".join(functions)
        elif query in functions:
            function_obj = getattr(self.jira, query)
            params = inspect.signature(function_obj).parameters
            return list(params.keys())
        else:
            return f"Function {query} not found."

    def run(self, mode: str, query: str) -> str:
        if mode == "jql":
            return self.search(query)
        elif mode == "project":
            return self.project()
        elif mode == "group":
            return self.group()
        elif mode == "group_users":
            return self.get_all_users_from_group(query)
        elif mode == "create_issue":
            return self.issue_create(query)
        elif mode == "other":
            return self.other(query)
        elif mode == "create_page":
            return self.page_create(query)
        elif mode == "get_jira_functions":
            return self.get_jira_functions(query)
        else:
            raise ValueError(f"Got unexpected mode {mode}")
