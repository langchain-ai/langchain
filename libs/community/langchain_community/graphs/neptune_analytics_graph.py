from typing import Any, Dict, Optional
import json
from .neptune_base_graph import NeptuneBaseGraph, NeptuneQueryException


class NeptuneAnalyticsGraph(NeptuneBaseGraph):
    """Neptune Analytics wrapper for graph operations.

    Args:
        client: optional boto3 Neptune client
        credentials_profile_name: optional AWS profile name
        region_name: optional AWS region, e.g., us-west-2
        graph_identifier: the graph identifier for a Neptune Analytics graph

    Example:
        .. code-block:: python

        graph = NeptuneAnalyticsGraph(
            graph_identifier='<my-graph-id>'
        )

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://python.langchain.com/docs/security for more information.
    """

    def __init__(
        self,
        graph_identifier: str,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
    ) -> None:
        """Create a new Neptune Analytics graph wrapper instance."""

        try:
            if client is not None:
                self.client = client
            else:
                import boto3

                if credentials_profile_name is not None:
                    session = boto3.Session(profile_name=credentials_profile_name)
                else:
                    # use default credentials
                    session = boto3.Session()

                self.graph_identifier = graph_identifier

                if region_name:
                    self.client = session.client(
                        "neptune-graph", region_name=region_name
                    )
                else:
                    self.client = session.client("neptune-graph")

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ModuleNotFoundError(
                    "NeptuneGraph requires a boto3 version 1.34.40 or greater."
                    "Please install it with `pip install -U boto3`."
                ) from e
            else:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        try:
            self._refresh_schema()
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "Could not get schema for Neptune database",
                    "detail": str(e),
                }
            )

    def query(self, query: str, params: dict = {}) -> Dict[str, Any]:
        """Query Neptune database."""
        try:
            resp = self.client.execute_query(
                graphIdentifier=self.graph_identifier,
                queryString=query,
                parameters=params,
                language="OPEN_CYPHER",
            )
            return json.loads(resp["payload"].read().decode("UTF-8"))["results"]
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "An error occurred while executing the query.",
                    "details": str(e),
                }
            )

    def _get_summary(self) -> Dict:

        try:
            response = self.client.get_graph_summary(
                graphIdentifier=self.graph_identifier, mode="detailed"
            )
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": ("Summary API error ocurred on Neptune Analytics"),
                    "details": str(e),
                }
            )

        try:
            summary = response["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary
