from typing import Optional

from langchain_core.callbacks import CallbackManagerForToolRun
from langchain_core.tools import BaseTool
from pydantic import Field

from langchain_community.utilities.atlassian import AtlassianAPIWrapper


class AtlassianAction(BaseTool):
    """Tool that queries the Atlassian API for both Jira and Confluence."""

    api_wrapper: AtlassianAPIWrapper = Field(default_factory=AtlassianAPIWrapper)
    mode: str
    name: str = ""
    description: str = ""

    def clean_query(self, query: str) -> str:
        """
        Remove invalid characters and validate query syntax for both JQL and CQL.

        Args:
            query (str): The query string to be cleaned and validated.

        Returns:
            str: The cleaned and validated query string.

        Raises:
            ValueError: If the query contains unmatched parentheses or is empty after cleaning.
        """
        # If query is empty, return it as is
        if not query:
            return query

        invalid_chars = ["`", "'"]
        for char in invalid_chars:
            query = query.replace(char, "")

        if query.count('"') % 2 != 0:
            query += '"'

        query = query.strip()

        if query.count("(") != query.count(")"):
            raise ValueError(f"Unmatched parentheses in query: {query}")

        if not query:
            raise ValueError("Query cannot be empty")

        return query

    def _run(
        self,
        instructions: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Use the Atlassian API to run an operation.

        Args:
            instructions (str): The instructions or query string to be executed.
            run_manager (Optional[CallbackManagerForToolRun]): Optional callback manager for the tool run.

        Returns:
            str: The result of the operation as a string. If there is a ValueError, the error message is returned.

        Raises:
            ValueError: If the query is invalid or has unmatched parentheses.
        """
        try:
            # Clean and validate the query
            cleaned_instructions = self.clean_query(instructions)

            # Run the cleaned query
            return self.api_wrapper.run(self.mode, cleaned_instructions)
        except ValueError as e:
            return str(e)
