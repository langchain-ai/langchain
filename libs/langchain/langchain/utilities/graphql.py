import json
from typing import Any, Callable, Dict, Optional

from pydantic import BaseModel, Extra, root_validator


class GraphQLAPIWrapper(BaseModel):
    """Wrapper around GraphQL API.

    To use, you should have the ``gql`` and ``graphql`` python packages installed.
    This wrapper will use the GraphQL API to conduct queries.
    """

    custom_headers: Optional[Dict[str, str]] = None
    graphql_endpoint: str
    gql_client: Any  #: :meta private:
    gql_function: Callable[[str], Any]  #: :meta private:
    custom_transport_auth: Any  #: :meta private:
    gql_schema: str
    auto_fetch_schema: bool

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator(pre=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that the python package exists in the environment.
        Pull graphql schema using introspection query if asked."""
        try:
            from gql import Client, gql
            from gql.transport.requests import RequestsHTTPTransport

        except ImportError as e:
            raise ImportError(
                "Could not import gql python package. "
                f"Try installing it with `pip install gql`. Received error: {e}"
            )
        headers = values.get("custom_headers", {})
        customAuth = values.get("custom_transport_auth", None)
        transport = RequestsHTTPTransport(
            url=values["graphql_endpoint"], headers=headers, auth=customAuth
        )

        client = Client(transport=transport, fetch_schema_from_transport=True)
        values["gql_client"] = client
        values["gql_function"] = gql
        values["gql_schema"] = ""

        return values

    def fetch_schema(self) -> str:
        try:
            from gql import gql
            from graphql import (
                GraphQLScalarType,
                build_client_schema,
                get_introspection_query,
            )
            from graphql.utilities.print_schema import print_filtered_schema

        except ImportError as e:
            raise ImportError(
                "Could not import gql or graphql python package. "
                f"Try installing it with `pip install gql graphql`. Received error: {e}"
            )

        query_intros = get_introspection_query(descriptions=True)
        document_node = gql(query_intros)
        intros_result: Any = self.gql_client.execute(document_node)

        # Removes introspection fields, directives, and scalars from schema
        # These are redundant and cause confusion with the model
        gql_schema = print_filtered_schema(
            build_client_schema(intros_result),
            directive_filter=lambda n: False,
            type_filter=lambda n: not (
                n.name.startswith("_") or isinstance(n, GraphQLScalarType)
            ),
        )

        return gql_schema

    def run(self, query: str) -> str:
        """Run a GraphQL query and get the results."""
        result = self._execute_query(query)
        return json.dumps(result, indent=2)

    def _execute_query(self, query: str) -> Dict[str, Any]:
        """Execute a GraphQL query and return the results."""
        document_node = self.gql_function(query)
        result = self.gql_client.execute(document_node)
        return result
