"""Test Neo4j functionality."""

from langchain_community.vectorstores.neo4j_vector import (
    dict_to_yaml_str,
    remove_lucene_chars,
)


def test_escaping_lucene() -> None:
    """Test escaping lucene characters"""
    assert remove_lucene_chars("Hello+World") == "Hello World"
    assert remove_lucene_chars("Hello World\\") == "Hello World"
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter!")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter&&")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("Bill&&Melinda Gates Foundation")
        == "Bill  Melinda Gates Foundation"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter(&&)")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter??")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter^")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter+")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter-")
        == "It is the end of the world. Take shelter"
    )
    assert (
        remove_lucene_chars("It is the end of the world. Take shelter~")
        == "It is the end of the world. Take shelter"
    )


def test_converting_to_yaml() -> None:
    example_dict = {
        "name": "John Doe",
        "age": 30,
        "skills": ["Python", "Data Analysis", "Machine Learning"],
        "location": {"city": "Ljubljana", "country": "Slovenia"},
    }

    yaml_str = dict_to_yaml_str(example_dict)

    expected_output = (
        "name: John Doe\nage: 30\nskills:\n- Python\n- "
        "Data Analysis\n- Machine Learning\nlocation:\n  city: Ljubljana\n"
        "  country: Slovenia\n"
    )

    assert yaml_str == expected_output
