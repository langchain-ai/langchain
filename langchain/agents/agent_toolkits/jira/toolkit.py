from typing import Dict, List

from langchain.agents.agent_toolkits.base import BaseToolkit
from langchain.tools import BaseTool
from langchain.tools.jira.prompt import (
    JIRA_CATCH_ALL_PROMPT,
    JIRA_CONFLUENCE_PAGE_CREATE_PROMPT,
    JIRA_GET_ALL_PROJECTS_PROMPT,
    JIRA_ISSUE_CREATE_PROMPT,
    JIRA_JQL_PROMPT,
)
from langchain.tools.jira.tool import JiraAction
from langchain.utilities.jira import JiraAPIWrapper


class JiraToolkit(BaseToolkit):
    """Jira Toolkit."""

    tools: List[BaseTool] = []

    @classmethod
    def from_jira_api_wrapper(cls, jira_api_wrapper: JiraAPIWrapper) -> "JiraToolkit":
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
            {
                "mode": "create_page",
                "name": "Create confluence page",
                "description": JIRA_CONFLUENCE_PAGE_CREATE_PROMPT,
            },
        ]
        tools = [
            JiraAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=jira_api_wrapper,
            )
            for action in operations
        ]
        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """Get the tools in the toolkit."""
        return self.tools
