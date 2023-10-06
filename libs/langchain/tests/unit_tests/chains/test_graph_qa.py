from typing import List

import pandas as pd

from langchain.chains.graph_qa.cypher import construct_schema, extract_cypher
from langchain.chains.graph_qa.cypher_utils import CypherQueryCorrector, Schema


def test_no_backticks() -> None:
    """Test if there are no backticks, so the original text should be returned."""
    query = "MATCH (n) RETURN n"
    output = extract_cypher(query)
    assert output == query


def test_backticks() -> None:
    """Test if there are backticks. Query from within backticks should be returned."""
    query = "You can use the following query: ```MATCH (n) RETURN n```"
    output = extract_cypher(query)
    assert output == "MATCH (n) RETURN n"


def test_exclude_types() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    exclude_types = ["Person", "DIRECTED"]
    output = construct_schema(structured_schema, [], exclude_types)
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "['(:Actor)-[:ACTED_IN]->(:Movie)']"
    )
    assert output == expected_schema


def test_include_types() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor", "ACTED_IN"]
    output = construct_schema(structured_schema, include_types, [])
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "['(:Actor)-[:ACTED_IN]->(:Movie)']"
    )
    assert output == expected_schema


def test_include_types2() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor"]
    output = construct_schema(structured_schema, include_types, [])
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "[]"
    )
    assert output == expected_schema


def test_include_types3() -> None:
    structured_schema = {
        "node_props": {
            "Movie": [{"property": "title", "type": "STRING"}],
            "Actor": [{"property": "name", "type": "STRING"}],
            "Person": [{"property": "name", "type": "STRING"}],
        },
        "rel_props": {},
        "relationships": [
            {"start": "Actor", "end": "Movie", "type": "ACTED_IN"},
            {"start": "Person", "end": "Movie", "type": "DIRECTED"},
        ],
    }
    include_types = ["Movie", "Actor", "ACTED_IN"]
    output = construct_schema(structured_schema, include_types, [])
    expected_schema = (
        "Node properties are the following: \n"
        " {'Movie': [{'property': 'title', 'type': 'STRING'}], "
        "'Actor': [{'property': 'name', 'type': 'STRING'}]}\n"
        "Relationships properties are the following: \n"
        " {}\nRelationships are: \n"
        "['(:Actor)-[:ACTED_IN]->(:Movie)']"
    )
    assert output == expected_schema


def test_validating_cypher_statements() -> None:
    cypher_file = "tests/unit_tests/data/cypher_corrector.csv"
    examples = pd.read_csv(cypher_file)
    examples.fillna("", inplace=True)
    for _, row in examples.iterrows():
        schema = load_schemas(row["schema"])
        corrector = CypherQueryCorrector(schema)
        assert corrector(row["statement"]) == row["correct_query"]


def load_schemas(str_schemas: str) -> List[Schema]:
    """
    Args:
        str_schemas: string of schemas
    """
    values = str_schemas.replace("(", "").replace(")", "").split(",")
    schemas = []
    for i in range(len(values) // 3):
        schemas.append(
            Schema(
                values[i * 3].strip(),
                values[i * 3 + 1].strip(),
                values[i * 3 + 2].strip(),
            )
        )
    return schemas
