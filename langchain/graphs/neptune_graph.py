import json
from typing import Any, Dict, List, Tuple, Union

import requests


class NeptuneQueryException(Exception):
    """A class to handle queries that fail to execute"""

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


class NeptuneGraph:
    """Neptune wrapper for graph operations. This version
    does not support Sigv4 signing of requests.

    Example:
        .. code-block:: python

        graph = NeptuneGraph(
            host='<my-cluster>',
            port=8182
        )
    """

    def __init__(self, host: str, port: int = 8182, use_https: bool = True) -> None:
        """Create a new Neptune graph wrapper instance."""

        if use_https:
            self.summary_url = (
                f"https://{host}:{port}/pg/statistics/summary?mode=detailed"
            )
            self.query_url = f"https://{host}:{port}/openCypher"
        else:
            self.summary_url = (
                f"http://{host}:{port}/pg/statistics/summary?mode=detailed"
            )
            self.query_url = f"http://{host}:{port}/openCypher"

        # Set schema
        try:
            self._refresh_schema()
        except NeptuneQueryException:
            raise ValueError("Could not get schema for Neptune database")

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Neptune database"""
        return self.schema

    def query(self, query: str, params: dict = {}) -> Dict[str, Any]:
        """Query Neptune database."""
        response = requests.post(url=self.query_url, data={"query": query})
        if response.ok:
            results = json.loads(response.content.decode())
            return results
        else:
            raise NeptuneQueryException(
                {
                    "message": "The generated query failed to execute",
                    "details": response.content.decode(),
                }
            )

    def _get_summary(self) -> Dict:
        response = requests.get(url=self.summary_url)
        if not response.ok:
            raise NeptuneQueryException(
                {
                    "message": (
                        "Summary API is not available for this instance of Neptune,"
                        "ensure the engine version is >=1.2.1.0"
                    ),
                    "details": response.content.decode(),
                }
            )
        try:
            summary = response.json()["payload"]["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary

    def _get_labels(self) -> Tuple[List[str], List[str]]:
        """Get node and edge labels from the Neptune statistics summary"""
        summary = self._get_summary()
        n_labels = summary["nodeLabels"]
        e_labels = summary["edgeLabels"]
        return n_labels, e_labels

    def _get_triples(self, e_labels: List[str]) -> List[str]:
        triple_query = """
        MATCH (a)-[e:{e_label}]->(b)
        WITH a,e,b LIMIT 3000
        RETURN DISTINCT labels(a) AS from, type(e) AS edge, labels(b) AS to
        LIMIT 10
        """

        triple_template = "(:{a})-[:{e}]->(:{b})"
        triple_schema = []
        for label in e_labels:
            q = triple_query.format(e_label=label)
            data = self.query(q)
            for d in data["results"]:
                triple = triple_template.format(
                    a=d["from"][0], e=d["edge"], b=d["to"][0]
                )
                triple_schema.append(triple)

        return triple_schema

    def _get_node_properties(self, n_labels: List[str], types: Dict) -> List:
        node_properties_query = """
        MATCH (a:{n_label})
        RETURN properties(a) AS props
        LIMIT 100
        """
        node_properties = []
        for label in n_labels:
            q = node_properties_query.format(n_label=label)
            data = {"label": label, "properties": self.query(q)["results"]}
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
        MATCH ()-[e:{e_label}]->()
        RETURN properties(e) AS props
        LIMIT 100
        """
        edge_properties = []
        for label in e_labels:
            q = edge_properties_query.format(e_label=label)
            data = {"label": label, "properties": self.query(q)["results"]}
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
