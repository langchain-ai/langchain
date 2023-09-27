from langchain.chains.graph_qa.cypher import construct_schema, extract_cypher


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
