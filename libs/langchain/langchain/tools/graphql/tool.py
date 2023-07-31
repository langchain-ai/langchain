import json

from typing import Optional, Dict, Any
from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.graphql import GraphQLAPIWrapper


class BaseGraphQLTool(BaseTool):
    """Base tool for querying a GraphQL API."""

    graphql_wrapper: GraphQLAPIWrapper

    name = "query_graphql"
    description = """\
    Input to this tool is a detailed and correct GraphQL query and optional query variables, graphql endpoint and request headers whereas output is the json format string of the result from the API.
    If the query is not correct, an error message will be returned.
    If an error is returned with 'Bad request' in it, rewrite the query and try again.
    If an error is returned with 'Unauthorized' in it, do not try again, but tell the user to change their authentication.

    Example Input: 
    query: 'query($name: String) {{ allUsers(name: $name) {{ id, name, email }} }}'
    query_variables: '{{"name": "John Wan"}}'
    graphql_endpoint: 'http://testserver/graphql'
    headers: '{{"Authorization": "Bearer testtoken", "Accept": "application/json"}}'

    """  # noqa: E501

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    def _run(
            self,
            query: str,
            query_variables: Optional[Dict[str, Any]] = None,
            graphql_endpoint: str = None,
            headers: Optional[Dict[str, str]] = None,
            run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        result = self.graphql_wrapper.run(query, query_variables, graphql_endpoint, headers)
        return json.dumps(result, indent=2)

    async def _arun(
            self,
            query: str,
            query_variables: Optional[Dict[str, Any]] = None,
            graphql_endpoint: str = None,
            headers: Optional[Dict[str, str]] = None,
            run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Use the Graphql tool asynchronously."""
        raise NotImplementedError("GraphQLAPIWrapper does not support async")
