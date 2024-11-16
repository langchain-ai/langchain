import re
import unittest
from collections import namedtuple
from typing import Any, Dict, List

from langchain_community.graphs.age_graph import AGEGraph


class TestAGEGraph(unittest.TestCase):
    def test_format_triples(self) -> None:
        test_input = [
            {"start": "from_a", "type": "edge_a", "end": "to_a"},
            {"start": "from_b", "type": "edge_b", "end": "to_b"},
        ]

        expected = [
            "(:`from_a`)-[:`edge_a`]->(:`to_a`)",
            "(:`from_b`)-[:`edge_b`]->(:`to_b`)",
        ]

        self.assertEqual(AGEGraph._format_triples(test_input), expected)

    def test_get_col_name(self) -> None:
        inputs = [
            ("a", 1),
            ("a as b", 1),
            (" c ", 1),
            (" c as d ", 1),
            ("sum(a)", 1),
            ("sum(a) as b", 1),
            ("count(*)", 1),
            ("count(*) as cnt", 1),
            ("true", 1),
            ("false", 1),
            ("null", 1),
        ]

        expected = [
            "a",
            "b",
            "c",
            "d",
            "sum_a",
            "b",
            "count_*",
            "cnt",
            "column_1",
            "column_1",
            "column_1",
        ]

        for idx, value in enumerate(inputs):
            self.assertEqual(AGEGraph._get_col_name(*value), expected[idx])

    def test_wrap_query(self) -> None:
        inputs = [
            # Positive case: Simple return clause
            """
            MATCH (keanu:Person {name:'Keanu Reeves'})
            RETURN keanu.name AS name, keanu.born AS born
            """,
            """
            MERGE (n:a {id: 1})
            """,
            # Negative case: Return in a string value
            """
            MATCH (n {description: "This will return a value"})
            MERGE (n)-[:RELATED]->(m)
            """,
            # Negative case: Return in a property key
            """
            MATCH (n {returnValue: "some value"})
            MERGE (n)-[:RELATED]->(m)
            """,
        ]

        expected = [
            # Expected output for the first positive case
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (keanu:Person {name:'Keanu Reeves'})
            RETURN keanu.name AS name, keanu.born AS born
            $$) AS (name agtype, born agtype);
            """,
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MERGE (n:a {id: 1})
            $$) AS (a agtype);
            """,
            # Expected output for the negative cases (no return clause)
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (n {description: "This will return a value"})
            MERGE (n)-[:RELATED]->(m)
            $$) AS (a agtype);
            """,
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (n {returnValue: "some value"})
            MERGE (n)-[:RELATED]->(m)
            $$) AS (a agtype);
            """,
        ]

        for idx, value in enumerate(inputs):
            self.assertEqual(
                re.sub(r"\s", "", AGEGraph._wrap_query(value, "test")),
                re.sub(r"\s", "", expected[idx]),
            )

        with self.assertRaises(ValueError):
            AGEGraph._wrap_query(
                """
            MATCH ()
            RETURN *
            """,
                "test",
            )

    def test_format_properties(self) -> None:
        inputs: List[Dict[str, Any]] = [{}, {"a": "b"}, {"a": "b", "c": 1, "d": True}]

        expected = ["{}", '{`a`: "b"}', '{`a`: "b", `c`: 1, `d`: true}']

        for idx, value in enumerate(inputs):
            self.assertEqual(AGEGraph._format_properties(value), expected[idx])

    def test_clean_graph_labels(self) -> None:
        inputs = ["label", "label 1", "label#$"]

        expected = ["label", "label_1", "label_"]

        for idx, value in enumerate(inputs):
            self.assertEqual(AGEGraph.clean_graph_labels(value), expected[idx])

    def test_record_to_dict(self) -> None:
        Record = namedtuple("Record", ["node1", "edge", "node2"])
        r = Record(
            node1='{"id": 1, "label": "label1", "properties":'
            + ' {"prop": "a"}}::vertex',
            edge='{"id": 3, "label": "edge", "end_id": 2, '
            + '"start_id": 1, "properties": {"test": "abc"}}::edge',
            node2='{"id": 2, "label": "label1", '
            + '"properties": {"prop": "b"}}::vertex',
        )

        result = AGEGraph._record_to_dict(r)

        expected = {
            "node1": {"prop": "a"},
            "edge": ({"prop": "a"}, "edge", {"prop": "b"}),
            "node2": {"prop": "b"},
        }

        self.assertEqual(result, expected)

        Record2 = namedtuple("Record2", ["string", "int", "float", "bool", "null"])
        r2 = Record2('"test"', "1", "1.5", "true", None)

        result = AGEGraph._record_to_dict(r2)

        expected2 = {
            "string": "test",
            "int": 1,
            "float": 1.5,
            "bool": True,
            "null": None,
        }

        self.assertEqual(result, expected2)
