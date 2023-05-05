from typing import Any, Dict, Optional
import json
from pydantic import BaseModel, Extra, root_validator

class GraphQLAPIWrapper(BaseModel):
    """Wrapper around GraphQL API.

    To use, you should have the ``gql`` python package installed.
    This wrapper will use the GraphQL API to conduct queries.
    """

    graphql_endpoint: str
    gql_client: Any  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment."""
        try:
            from gql import gql, Client
            from gql.transport.requests import RequestsHTTPTransport

            transport = RequestsHTTPTransport(url=values["graphql_endpoint"])
            client = Client(transport=transport, fetch_schema_from_transport=True)
            values["gql_client"] = client
        except ImportError:
            raise ValueError(
                "Could not import gql python package. "
                "Please install it with `pip install gql`."
            )
        return values

    def run(self, query: str) -> str:
        """Run a GraphQL query and get the results."""
        result = self._execute_query(query)
        return json.dumps(result, indent=2)

    def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a GraphQL query and return the results."""
        query = gql(query)
        result = self.gql_client.execute(query)
        return result
