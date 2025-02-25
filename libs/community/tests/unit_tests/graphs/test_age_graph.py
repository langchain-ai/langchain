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
        """Test basic query wrapping functionality."""
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
            # Second test case (no RETURN clause)
            """
                SELECT * FROM ag_catalog.cypher('test', $$
                MERGE (n:a {id: 1})
                $$) AS (a agtype);
                """,
            # Expected output for the negative cases (no RETURN clause)
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
            result = AGEGraph._wrap_query(value, "test")
            expected_result = expected[idx]
            self.assertEqual(
                re.sub(r"\s", "", result),
                re.sub(r"\s", "", expected_result),
                (
                    f"Failed on test case {idx + 1}\n"
                    f"Input:\n{value}\n"
                    f"Expected:\n{expected_result}\n"
                    f"Got:\n{result}"
                ),
            )

    def test_wrap_query_union_except(self) -> None:
        """Test query wrapping with UNION and EXCEPT operators."""
        inputs = [
            # UNION case
            """
                MATCH (n:Person)
                RETURN n.name AS name, n.age AS age
                UNION
                MATCH (n:Employee)
                RETURN n.name AS name, n.salary AS salary
                """,
            """
                MATCH (a:Employee {name: "Alice"})
                RETURN a.name AS name
                UNION
                MATCH (b:Manager {name: "Bob"})
                RETURN b.name AS name
                """,
            # Complex UNION case
            """
                MATCH (n)-[r]->(m)
                RETURN n.name AS source, type(r) AS relationship, m.name AS target
                UNION
                MATCH (m)-[r]->(n)
                RETURN m.name AS source, type(r) AS relationship, n.name AS target
                """,
            """
                MATCH (a:Person)-[:FRIEND]->(b:Person)
                WHERE a.age > 30
                RETURN a.name AS name
                UNION
                MATCH (c:Person)-[:FRIEND]->(d:Person)
                WHERE c.age < 25
                RETURN c.name AS name
                """,
            # EXCEPT case
            """
                MATCH (n:Person)
                RETURN n.name AS name
                EXCEPT
                MATCH (n:Employee)
                RETURN n.name AS name
                """,
            """
                MATCH (a:Person)
                RETURN a.name AS name, a.age AS age
                EXCEPT
                MATCH (b:Person {name: "Alice", age: 30})
                RETURN b.name AS name, b.age AS age   
                """,
        ]

        expected = [
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (n:Person)
            RETURN n.name AS name, n.age AS age
            UNION
            MATCH (n:Employee)
            RETURN n.name AS name, n.salary AS salary
            $$) AS (name agtype, age agtype, salary agtype);
            """,
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (a:Employee {name: "Alice"})
            RETURN a.name AS name
            UNION
            MATCH (b:Manager {name: "Bob"})
            RETURN b.name AS name
            $$) AS (name agtype);  
            """,
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (n)-[r]->(m)
            RETURN n.name AS source, type(r) AS relationship, m.name AS target
            UNION
            MATCH (m)-[r]->(n)
            RETURN m.name AS source, type(r) AS relationship, n.name AS target
            $$) AS (source agtype, relationship agtype, target agtype);
            """,
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (a:Person)-[:FRIEND]->(b:Person)
            WHERE a.age > 30
            RETURN a.name AS name
            UNION
            MATCH (c:Person)-[:FRIEND]->(d:Person)
            WHERE c.age < 25
            RETURN c.name AS name
            $$) AS (name agtype);
            """,
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (n:Person)
            RETURN n.name AS name
            EXCEPT
            MATCH (n:Employee)
            RETURN n.name AS name
            $$) AS (name agtype);
            """,
            """
            SELECT * FROM ag_catalog.cypher('test', $$
            MATCH (a:Person)
            RETURN a.name AS name, a.age AS age
            EXCEPT
            MATCH (b:Person {name: "Alice", age: 30})
            RETURN b.name AS name, b.age AS age
            $$) AS (name agtype, age agtype);
            """,
        ]

        for idx, value in enumerate(inputs):
            result = AGEGraph._wrap_query(value, "test")
            expected_result = expected[idx]
            self.assertEqual(
                re.sub(r"\s", "", result),
                re.sub(r"\s", "", expected_result),
                (
                    f"Failed on test case {idx + 1}\n"
                    f"Input:\n{value}\n"
                    f"Expected:\n{expected_result}\n"
                    f"Got:\n{result}"
                ),
            )

    def test_wrap_query_errors(self) -> None:
        """Test error cases for query wrapping."""
        error_cases = [
            # Empty query
            "",
            # Return * case
            """
            MATCH ()
            RETURN *
            """,
            # Return * in UNION
            """
            MATCH (n:Person)
            RETURN n.name
            UNION
            MATCH ()
            RETURN *
            """,
        ]

        for query in error_cases:
            with self.assertRaises(ValueError):
                AGEGraph._wrap_query(query, "test")

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
