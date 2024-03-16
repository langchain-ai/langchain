from abc import abstractmethod
from typing import Any, Dict, List, Tuple, Union


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


class NeptuneBaseGraph:
    """This is an abstract base class that represents the shared features across the NeptuneGraph and NeptuneAnalyticsGraph classes."""

    def __init__():
        pass

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Neptune database"""
        return self.schema

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> Dict[str, Any]:
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
