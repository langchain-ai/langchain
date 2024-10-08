from typing import Dict, List

from langchain_core.tools import BaseToolkit

from langchain_community.tools import BaseTool
from langchain_community.tools.atlassian.prompt import (
    ATLASSIAN_CONFLUENCE_CATCH_ALL_PROMPT,
    ATLASSIAN_CONFLUENCE_CQL_PROMPT,
    ATLASSIAN_CONFLUENCE_GET_FUNCTIONS_PROMPT,
    ATLASSIAN_JIRA_CATCH_ALL_PROMPT,
    ATLASSIAN_JIRA_GET_FUNCTIONS_PROMPT,
    ATLASSIAN_JIRA_JQL_PROMPT,
)
from langchain_community.tools.atlassian.tool import AtlassianAction
from langchain_community.utilities.atlassian import AtlassianAPIWrapper


class AtlassianToolkit(BaseToolkit):
    """Atlassian Toolkit.

    *Security Note*: This toolkit contains tools that can read and modify
        the state of a service; e.g., by creating, deleting, or updating,
        reading underlying data.

        See https://python.langchain.com/docs/security,
        https://developer.atlassian.com/cloud/jira/software/security-overview/ for more information.

    Parameters:
        tools: Dict[str, List[BaseTool]]. The tools in the toolkit grouped by solution type.
        Default is an empty dictionary.
    """

    tools: List[BaseTool] = []

    @classmethod
    def from_atlassian_api_wrapper(
        cls, atlassian_api_wrapper: AtlassianAPIWrapper
    ) -> "AtlassianToolkit":
        """Create an AtlassianToolkit from an AtlassianAPIWrapper.

        Args:
            atlassian_api_wrapper: AtlassianAPIWrapper. The Atlassian API wrapper.

        Returns:
            AtlassianToolkit. The Atlassian toolkit.

        See https://atlassian-python-api.readthedocs.io/ for more information.
        """
        jira_operations: List[Dict] = [
            {
                "mode": "jira_jql",
                "name": "JQL Query",
                "description": ATLASSIAN_JIRA_JQL_PROMPT,
            },
            {
                "mode": "jira_other",
                "name": "Catch all Jira API call",
                "description": ATLASSIAN_JIRA_CATCH_ALL_PROMPT,
            },
            {
                "mode": "jira_get_functions",
                "name": "Get all Jira API Functions",
                "description": ATLASSIAN_JIRA_GET_FUNCTIONS_PROMPT,
            },
        ]

        confluence_operations: List[Dict] = [
            {
                "mode": "confluence_cql",
                "name": "CQL Query",
                "description": ATLASSIAN_CONFLUENCE_CQL_PROMPT,
            },
            {
                "mode": "confluence_other",
                "name": "Catch all Confluence API call",
                "description": ATLASSIAN_CONFLUENCE_CATCH_ALL_PROMPT,
            },
            {
                "mode": "confluence_get_functions",
                "name": "Get all Confluence API Functions",
                "description": ATLASSIAN_CONFLUENCE_GET_FUNCTIONS_PROMPT,
            },
        ]

        jira_tools = [
            AtlassianAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=atlassian_api_wrapper,
            )
            for action in jira_operations
        ]

        confluence_tools = [
            AtlassianAction(
                name=action["name"],
                description=action["description"],
                mode=action["mode"],
                api_wrapper=atlassian_api_wrapper,
            )
            for action in confluence_operations
        ]

        tools = jira_tools + confluence_tools

        return cls(tools=tools)

    def get_tools(self) -> List[BaseTool]:
        """
        Get all the tools in the toolkit, grouped by solution type.

        Returns:
            List[BaseTool]: A list of all tools in the toolkit.
        """
        return self.tools
