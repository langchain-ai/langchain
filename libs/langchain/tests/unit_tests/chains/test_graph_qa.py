from langchain.chains.graph_qa.cypher import extract_cypher


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
