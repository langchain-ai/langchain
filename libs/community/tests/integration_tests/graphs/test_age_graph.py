import os
import re
import unittest
from typing import Any, Dict

from langchain_core.documents import Document

from langchain_community.graphs.age_graph import AGEGraph
from langchain_community.graphs.graph_document import GraphDocument, Node, Relationship

test_data = [
    GraphDocument(
        nodes=[
            Node(id="foo", type="foo"),
            Node(id="bar", type="bar"),
            Node(id="foo", type="foo", properties={"property_a": "a"}),
        ],
        relationships=[
            Relationship(
                source=Node(id="foo", type="foo"),
                target=Node(id="bar", type="bar"),
                type="REL",
            )
        ],
        source=Document(page_content="source document"),
    )
]


class TestAGEGraph(unittest.TestCase):
    def test_node_properties(self) -> None:
        conf = {
            "database": os.getenv("AGE_PGSQL_DB"),
            "user": os.getenv("AGE_PGSQL_USER"),
            "password": os.getenv("AGE_PGSQL_PASSWORD"),
            "host": os.getenv("AGE_PGSQL_HOST", "localhost"),
            "port": int(os.getenv("AGE_PGSQL_PORT", 5432)),
        }

        self.assertIsNotNone(conf["database"])
        self.assertIsNotNone(conf["user"])
        self.assertIsNotNone(conf["password"])

        graph_name = os.getenv("AGE_GRAPH_NAME", "age_test")

        graph = AGEGraph(graph_name, conf)

        graph.query("MATCH (n) DETACH DELETE n")

        # Create two nodes and a relationship
        graph.query(
            """
            CREATE (la:LabelA {property_a: 'a'})
            CREATE (lb:LabelB)
            CREATE (lc:LabelC)
            MERGE (la)-[:REL_TYPE]-> (lb)
            MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
            """
        )
        # Refresh schema information
        # graph.refresh_schema()

        n_labels, e_labels = graph._get_labels()

        node_properties = graph._get_node_properties(n_labels)

        expected_node_properties = [
            {
                "properties": [{"property": "property_a", "type": "STRING"}],
                "labels": "LabelA",
            },
            {
                "properties": [],
                "labels": "LabelB",
            },
            {
                "properties": [],
                "labels": "LabelC",
            },
        ]

        self.assertEqual(
            sorted(node_properties, key=lambda x: x["labels"]), expected_node_properties
        )

    def test_edge_properties(self) -> None:
        conf = {
            "database": os.getenv("AGE_PGSQL_DB"),
            "user": os.getenv("AGE_PGSQL_USER"),
            "password": os.getenv("AGE_PGSQL_PASSWORD"),
            "host": os.getenv("AGE_PGSQL_HOST", "localhost"),
            "port": int(os.getenv("AGE_PGSQL_PORT", 5432)),
        }

        self.assertIsNotNone(conf["database"])
        self.assertIsNotNone(conf["user"])
        self.assertIsNotNone(conf["password"])

        graph_name = os.getenv("AGE_GRAPH_NAME", "age_test")

        graph = AGEGraph(graph_name, conf)

        graph.query("MATCH (n) DETACH DELETE n")
        # Create two nodes and a relationship
        graph.query(
            """
            CREATE (la:LabelA {property_a: 'a'})
            CREATE (lb:LabelB)
            CREATE (lc:LabelC)
            MERGE (la)-[:REL_TYPE]-> (lb)
            MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
            """
        )
        # Refresh schema information
        # graph.refresh_schema()

        n_labels, e_labels = graph._get_labels()

        relationships_properties = graph._get_edge_properties(e_labels)

        expected_relationships_properties = [
            {
                "type": "REL_TYPE",
                "properties": [{"property": "rel_prop", "type": "STRING"}],
            }
        ]

        self.assertEqual(relationships_properties, expected_relationships_properties)

    def test_relationships(self) -> None:
        conf = {
            "database": os.getenv("AGE_PGSQL_DB"),
            "user": os.getenv("AGE_PGSQL_USER"),
            "password": os.getenv("AGE_PGSQL_PASSWORD"),
            "host": os.getenv("AGE_PGSQL_HOST", "localhost"),
            "port": int(os.getenv("AGE_PGSQL_PORT", 5432)),
        }

        self.assertIsNotNone(conf["database"])
        self.assertIsNotNone(conf["user"])
        self.assertIsNotNone(conf["password"])

        graph_name = os.getenv("AGE_GRAPH_NAME", "age_test")

        graph = AGEGraph(graph_name, conf)

        graph.query("MATCH (n) DETACH DELETE n")
        # Create two nodes and a relationship
        graph.query(
            """
            CREATE (la:LabelA {property_a: 'a'})
            CREATE (lb:LabelB)
            CREATE (lc:LabelC)
            MERGE (la)-[:REL_TYPE]-> (lb)
            MERGE (la)-[:REL_TYPE {rel_prop: 'abc'}]-> (lc)
            """
        )
        # Refresh schema information
        # graph.refresh_schema()

        n_labels, e_labels = graph._get_labels()

        relationships = graph._get_triples(e_labels)

        expected_relationships = [
            {"start": "LabelA", "type": "REL_TYPE", "end": "LabelB"},
            {"start": "LabelA", "type": "REL_TYPE", "end": "LabelC"},
        ]

        self.assertEqual(
            sorted(relationships, key=lambda x: x["end"]), expected_relationships
        )

    def test_add_documents(self) -> None:
        conf = {
            "database": os.getenv("AGE_PGSQL_DB"),
            "user": os.getenv("AGE_PGSQL_USER"),
            "password": os.getenv("AGE_PGSQL_PASSWORD"),
            "host": os.getenv("AGE_PGSQL_HOST", "localhost"),
            "port": int(os.getenv("AGE_PGSQL_PORT", 5432)),
        }

        self.assertIsNotNone(conf["database"])
        self.assertIsNotNone(conf["user"])
        self.assertIsNotNone(conf["password"])

        graph_name = os.getenv("AGE_GRAPH_NAME", "age_test")

        graph = AGEGraph(graph_name, conf)

        # Delete all nodes in the graph
        graph.query("MATCH (n) DETACH DELETE n")
        # Create two nodes and a relationship
        graph.add_graph_documents(test_data)
        output = graph.query(
            "MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY labels(n)"
        )
        self.assertEqual(
            output, [{"label": ["bar"], "count": 1}, {"label": ["foo"], "count": 1}]
        )

    def test_add_documents_source(self) -> None:
        conf = {
            "database": os.getenv("AGE_PGSQL_DB"),
            "user": os.getenv("AGE_PGSQL_USER"),
            "password": os.getenv("AGE_PGSQL_PASSWORD"),
            "host": os.getenv("AGE_PGSQL_HOST", "localhost"),
            "port": int(os.getenv("AGE_PGSQL_PORT", 5432)),
        }

        self.assertIsNotNone(conf["database"])
        self.assertIsNotNone(conf["user"])
        self.assertIsNotNone(conf["password"])

        graph_name = os.getenv("AGE_GRAPH_NAME", "age_test")

        graph = AGEGraph(graph_name, conf)

        # Delete all nodes in the graph
        graph.query("MATCH (n) DETACH DELETE n")
        # Create two nodes and a relationship
        graph.add_graph_documents(test_data, include_source=True)
        output = graph.query(
            "MATCH (n) RETURN labels(n) AS label, count(*) AS count ORDER BY labels(n)"
        )

        expected = [
            {"label": ["bar"], "count": 1},
            {"label": ["Document"], "count": 1},
            {"label": ["foo"], "count": 1},
        ]
        self.assertEqual(output, expected)

    def test_get_schema(self) -> None:
        conf = {
            "database": os.getenv("AGE_PGSQL_DB"),
            "user": os.getenv("AGE_PGSQL_USER"),
            "password": os.getenv("AGE_PGSQL_PASSWORD"),
            "host": os.getenv("AGE_PGSQL_HOST", "localhost"),
            "port": int(os.getenv("AGE_PGSQL_PORT", 5432)),
        }

        self.assertIsNotNone(conf["database"])
        self.assertIsNotNone(conf["user"])
        self.assertIsNotNone(conf["password"])

        graph_name = os.getenv("AGE_GRAPH_NAME", "age_test")

        graph = AGEGraph(graph_name, conf)

        graph.query("MATCH (n) DETACH DELETE n")

        graph.refresh_schema()

        expected = """
            Node properties are the following:
            []
            Relationship properties are the following:
            []
            The relationships are the following:
            []
            """
        # check that works on empty schema
        self.assertEqual(
            re.sub(r"\s", "", graph.get_schema), re.sub(r"\s", "", expected)
        )

        expected_structured: Dict[str, Any] = {
            "node_props": {},
            "rel_props": {},
            "relationships": [],
            "metadata": {},
        }

        self.assertEqual(graph.get_structured_schema, expected_structured)

        # Create two nodes and a relationship
        graph.query(
            """
            MERGE (a:a {id: 1})-[b:b {id: 2}]-> (c:c {id: 3})
            """
        )

        # check that schema doesn't update without refresh
        self.assertEqual(
            re.sub(r"\s", "", graph.get_schema), re.sub(r"\s", "", expected)
        )
        self.assertEqual(graph.get_structured_schema, expected_structured)

        # two possible orderings of node props
        expected_possibilities = [
            """
            Node properties are the following:
            [
                {'properties': [{'property': 'id', 'type': 'INTEGER'}], 'labels': 'a'},
                {'properties': [{'property': 'id', 'type': 'INTEGER'}], 'labels': 'c'}
            ]
            Relationship properties are the following:
            [
                {'properties': [{'property': 'id', 'type': 'INTEGER'}], 'type': 'b'}
            ]
            The relationships are the following:
            [
                '(:`a`)-[:`b`]->(:`c`)'
            ]
            """,
            """
            Node properties are the following:
            [
                {'properties': [{'property': 'id', 'type': 'INTEGER'}], 'labels': 'c'},
                {'properties': [{'property': 'id', 'type': 'INTEGER'}], 'labels': 'a'}
            ]
            Relationship properties are the following:
            [
                {'properties': [{'property': 'id', 'type': 'INTEGER'}], 'type': 'b'}
            ]
            The relationships are the following:
            [
                '(:`a`)-[:`b`]->(:`c`)'
            ]
            """,
        ]

        expected_structured2 = {
            "node_props": {
                "a": [{"property": "id", "type": "INTEGER"}],
                "c": [{"property": "id", "type": "INTEGER"}],
            },
            "rel_props": {"b": [{"property": "id", "type": "INTEGER"}]},
            "relationships": [{"start": "a", "type": "b", "end": "c"}],
            "metadata": {},
        }

        graph.refresh_schema()

        # check that schema is refreshed
        self.assertIn(
            re.sub(r"\s", "", graph.get_schema),
            [re.sub(r"\s", "", x) for x in expected_possibilities],
        )
        self.assertEqual(graph.get_structured_schema, expected_structured2)
