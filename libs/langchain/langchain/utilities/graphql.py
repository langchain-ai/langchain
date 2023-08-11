import json
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Extra, root_validator


class GraphQLAPIWrapper(BaseModel):
    """Wrapper around GraphQL API.

    To use, you should have the ``gql`` python package installed.
    This wrapper will use the GraphQL API to conduct queries.
    """

    custom_headers: Optional[Dict[str, str]] = None
    graphql_endpoint: str
    gql_client: Any  #: :meta private:
    gql_function: Callable[[str], Any]  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            from gql import Client, gql
            from gql.transport.requests import RequestsHTTPTransport
        except ImportError as e:
            raise ImportError(
                "Could not import gql python package. "
                f"Try installing it with `pip install gql`. Received error: {e}"
            )
        headers = values.get("custom_headers")
        transport = RequestsHTTPTransport(
            url=values["graphql_endpoint"],
            headers=headers,
        )
        client = Client(transport=transport, fetch_schema_from_transport=True)
        values["gql_client"] = client
        values["gql_function"] = gql
        return values

    def run(self, query: str) -> str:
        """Run a GraphQL query and get the results."""
        result = self._execute_query(query)
        return json.dumps(result, indent=2)

    def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a GraphQL query and return the results."""
        document_node = self.gql_function(query)
        result = self.gql_client.execute(document_node)
        return result
