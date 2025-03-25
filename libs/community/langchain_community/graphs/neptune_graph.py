import json
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core._api.deprecation import deprecated


class NeptuneQueryException(Exception):
    """Exception for the Neptune queries."""

    def __init__(self, exception: Union[str, Dict]):
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


class BaseNeptuneGraph(ABC):
    """Abstract base class for Neptune."""

    @property
    def get_schema(self) -> str:
        """Return the schema of the Neptune database"""
        return self.schema

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> dict:
        raise NotImplementedError()

    @abstractmethod
    def _get_summary(self) -> Dict:
        raise NotImplementedError()

    def _get_labels(self) -> Tuple[List[str], List[str]]:
        """Get node and edge labels from the Neptune statistics summary"""
        summary = self._get_summary()
        n_labels = summary["nodeLabels"]
        e_labels = summary["edgeLabels"]
        return n_labels, e_labels

    def _get_triples(self, e_labels: List[str]) -> List[str]:
        triple_query = """
        MATCH (a)-[e:`{e_label}`]->(b)
        WITH a,e,b LIMIT 3000
        RETURN DISTINCT labels(a) AS from, type(e) AS edge, labels(b) AS to
        LIMIT 10
        """

        triple_template = "(:`{a}`)-[:`{e}`]->(:`{b}`)"
        triple_schema = []
        for label in e_labels:
            q = triple_query.format(e_label=label)
            data = self.query(q)
            for d in data:
                triple = triple_template.format(
                    a=d["from"][0], e=d["edge"], b=d["to"][0]
                )
                triple_schema.append(triple)

        return triple_schema

    def _get_node_properties(self, n_labels: List[str], types: Dict) -> List:
        node_properties_query = """
        MATCH (a:`{n_label}`)
        RETURN properties(a) AS props
        LIMIT 100
        """
        node_properties = []
        for label in n_labels:
            q = node_properties_query.format(n_label=label)
            data = {"label": label, "properties": self.query(q)}
            s = set({})
            for p in data["properties"]:
                for k, v in p["props"].items():
                    s.add((k, types[type(v).__name__]))

            np = {
                "properties": [{"property": k, "type": v} for k, v in s],
                "labels": label,
            }
            node_properties.append(np)

        return node_properties

    def _get_edge_properties(self, e_labels: List[str], types: Dict[str, Any]) -> List:
        edge_properties_query = """
        MATCH ()-[e:`{e_label}`]->()
        RETURN properties(e) AS props
        LIMIT 100
        """
        edge_properties = []
        for label in e_labels:
            q = edge_properties_query.format(e_label=label)
            data = {"label": label, "properties": self.query(q)}
            s = set({})
            for p in data["properties"]:
                for k, v in p["props"].items():
                    s.add((k, types[type(v).__name__]))

            ep = {
                "type": label,
                "properties": [{"property": k, "type": v} for k, v in s],
            }
            edge_properties.append(ep)

        return edge_properties

    def _refresh_schema(self) -> None:
        """
        Refreshes the Neptune graph schema information.
        """

        types = {
            "str": "STRING",
            "float": "DOUBLE",
            "int": "INTEGER",
            "list": "LIST",
            "dict": "MAP",
            "bool": "BOOLEAN",
        }
        n_labels, e_labels = self._get_labels()
        triple_schema = self._get_triples(e_labels)
        node_properties = self._get_node_properties(n_labels, types)
        edge_properties = self._get_edge_properties(e_labels, types)

        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {triple_schema}
        """


@deprecated(
    since="0.3.15",
    removal="1.0",
    alternative_import="langchain_aws.NeptuneAnalyticsGraph",
)
class NeptuneAnalyticsGraph(BaseNeptuneGraph):
    """Neptune Analytics wrapper for graph operations.

    Parameters:
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
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ImportError(
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
                    "message": ("Summary API error occurred on Neptune Analytics"),
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


@deprecated(
    since="0.3.15",
    removal="1.0",
    alternative_import="langchain_aws.NeptuneGraph",
)
class NeptuneGraph(BaseNeptuneGraph):
    """Neptune wrapper for graph operations.

    Parameters:
        host: endpoint for the database instance
        port: port number for the database instance, default is 8182
        use_https: whether to use secure connection, default is True
        client: optional boto3 Neptune client
        credentials_profile_name: optional AWS profile name
        region_name: optional AWS region, e.g., us-west-2
        sign: optional, whether to sign the request payload, default is True

    Example:
        .. code-block:: python

        graph = NeptuneGraph(
            host='<my-cluster>',
            port=8182
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
        host: str,
        port: int = 8182,
        use_https: bool = True,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        sign: bool = True,
    ) -> None:
        """Create a new Neptune graph wrapper instance."""

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

                client_params = {}
                if region_name:
                    client_params["region_name"] = region_name

                protocol = "https" if use_https else "http"

                client_params["endpoint_url"] = f"{protocol}://{host}:{port}"

                if sign:
                    self.client = session.client("neptunedata", **client_params)
                else:
                    from botocore import UNSIGNED
                    from botocore.config import Config

                    self.client = session.client(
                        "neptunedata",
                        **client_params,
                        config=Config(signature_version=UNSIGNED),
                    )

        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ImportError(
                    "NeptuneGraph requires a boto3 version 1.28.38 or greater."
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
            return self.client.execute_open_cypher_query(openCypherQuery=query)[
                "results"
            ]
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "An error occurred while executing the query.",
                    "details": str(e),
                }
            )

    def _get_summary(self) -> Dict:
        try:
            response = self.client.get_propertygraph_summary()
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": (
                        "Summary API is not available for this instance of Neptune,"
                        "ensure the engine version is >=1.2.1.0"
                    ),
                    "details": str(e),
                }
            )

        try:
            summary = response["payload"]["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary
