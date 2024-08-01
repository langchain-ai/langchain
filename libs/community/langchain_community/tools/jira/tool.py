from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.pydantic_v1 import Field
from langchain_core.tools import BaseTool

from langchain_community.utilities.jira import JiraAPIWrapper


class JiraAction(BaseTool):
    """Tool that queries the Atlassian Jira API."""

    api_wrapper: JiraAPIWrapper = Field(default_factory=JiraAPIWrapper)  # type: ignore[arg-type]
    mode: str
    name: str = ""
    description: str = ""

    def clean_jql_query(self, query: str) -> str:
        """Remove invalid characters and validate JQL query syntax."""
        # Remove backticks and other invalid characters
        invalid_chars = ['`', "'"]
        for char in invalid_chars:
            query = query.replace(char, '')

        # Ensure double quotes are properly closed
        if query.count('"') % 2 != 0:
            raise ValueError("Unmatched double quotes in JQL query: " + query)

        return query

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Atlassian Jira API to run an operation."""
        try:
            # Clean and validate the JQL query
            cleaned_instructions = self.clean_jql_query(instructions)

            # Run the cleaned JQL query
            return self.api_wrapper.run(self.mode, cleaned_instructions)
        except ValueError as e:
            return str(e)
