import json
from typing import Any, Dict, Optional
from gql import gql, Client
from gql.transport.requests import RequestsHTTPTransport

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool


class BaseGraphQLTool(BaseTool):
    """Base tool for querying a GraphQL API."""

    graphql_endpoint: str

    name = "query_graphql"
    description = """
    Input to this tool is a detailed and correct GraphQL query, output is a result from the API.
    If the query is not correct, an error message will be returned.
    If an error is returned with 'Bad request' in it, rewrite the query and try again.
    If an error is returned with 'Unauthorized' in it, do not try again, but tell the user to change their authentication.

    Example Input: 'query { allUsers { id, name, email } }'
    """ # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def get_client(self):
        """Create and return a GraphQL client."""
        transport = RequestsHTTPTransport(url=self.graphql_endpoint)
        client = Client(transport=transport, fetch_schema_from_transport=True)
        return client

    def _run_query(self, query: str) -> Dict[str, Any]:
        """Execute a GraphQL query and return the results."""
        client = self.get_client()
        query = gql(query)
        result = client.execute(query)
        return result
 
    def _run(
        self,
        tool_input: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self._run_query(tool_input)
        return json.dumps(result, indent=2)
'''
    async def _arun(
        self,
        tool_input: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        # For now, we use the synchronous _run_query method as there isn't
        # an async version available in the gql library.
        result = self._run_query(tool_input)
        return json.dumps(result, indent=2)
'''