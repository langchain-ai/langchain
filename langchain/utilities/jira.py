"""Util that calls Jira."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Extra, root_validator

from langchain.utils import get_from_dict_or_env

from langchain.tools.jira.prompt import JIRA_JQL_PROMPT, JIRA_CREATE_PROMPT, JIRA_CATCH_ALL_PROMPT

import json

# todo: add update and delete operations, add jql error back to response, solve pagination issue
class JiraAPIWrapper(BaseModel):
    """Wrapper for Jira API."""

    jira: Any  #: :meta private:
    jira_username: Optional[str] = None
    jira_api_token: Optional[str] = None
    jira_instance_url: Optional[str] = None

    operations: List[str] = [
        {
            "id": "jql",
            "name": "JQL",
            "description": JIRA_JQL_PROMPT,
        },
        {
            "id": "create",
            "name": "Create Issue",
            "description": JIRA_CREATE_PROMPT,
        },
        {
            "id": "other",
            "name": "All jira operations other than create and jql",
            "description": JIRA_CATCH_ALL_PROMPT,
        }
    ]

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    def list(self) -> List[str]:
        # todo make this a list of dicts with name and description
        return self.operations

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        jira_username = get_from_dict_or_env(values, "jira_username", "JIRA_USERNAME")
        values["jira_username"] = jira_username

        jira_api_token = get_from_dict_or_env(values, "jira_api_token", "JIRA_API_TOKEN")
        values["jira_api_token"] = jira_api_token

        jira_instance_url = get_from_dict_or_env(values, "jira_instance_url", "JIRA_INSTANCE_URL")
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
            cloud=True)
        values["jira"] = jira

        return values

    def jql_results_to_text(self, issues):
        raw_string = json.dumps(issues, indent=4)
        # Start string
        parsed_string = "Returned issues:\n\n"
        # Process json
        for issue in issues["issues"]:
            # Simple fields
            key = issue["key"]
            summary = issue["fields"]["summary"]
            created = issue["fields"]["created"][0:10]
            priority = issue["fields"]["priority"]["name"]
            status = issue["fields"]["status"]["name"]
            try:
                assignee = issue["fields"]["assignee"]["displayName"]
            except:
                assignee = "None"
            # Related issues
            rel_issues = ""
            for related_issue in issue["fields"]["issuelinks"]:
                if "inwardIssue" in related_issue.keys():
                    rel_type = related_issue["type"]["inward"]
                    rel_key = related_issue["inwardIssue"]["key"]
                    rel_summary = related_issue["inwardIssue"]["fields"]["summary"]
                if "outwardIssue" in related_issue.keys():
                    rel_type = related_issue["type"]["outward"]
                    rel_key = related_issue["outwardIssue"]["key"]
                    rel_summary = related_issue["outwardIssue"]["fields"]["summary"]
                rel_issues += f"""        {rel_type} {rel_key} {rel_summary}"""
            # Add text
            parsed_string += f"""{key}: {summary}\n    Created on: {created}\n    Assignee: {assignee}\n    Priority: {priority}\n    Status: {status}\n{rel_issues}\n\n"""
        # Return parsed string
        return parsed_string, raw_string

    def search(self, query: str) -> str:
        jql_response = self.jira.jql(query)
        parsed, raw = self.jql_results_to_text(jql_response)
        return parsed

    def create(self, query: str) -> str:
        try:
            params = json.loads(query)
            print(params)
            self.jira.create_issue(fields=dict(params))
        except ValueError as e:
            print("Error: JSON provided by LLM incorrect")
            print(e)

    def other(self, query: str) -> str:
        try:
            exec(query)
        except ValueError as e:
            print("Error: jira call provided by LLM incorrect")
            print(e)

    def run(self, mode: str, query: str) -> str:
        """Run query through Jira and parse result."""
        if mode == "jql":
            return self.search(query)
        elif mode == "create":
            return self.create(query)
        elif mode == "other":
            return self.other(query)