import pytest

from langchain_community.graphs import OntotextGraphDBGraph

"""
cd libs/community/tests/integration_tests/graphs/docker-compose-ontotext-graphdb
./start.sh
"""


def test_query_method_with_valid_query() -> None:
    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/langchain",
    )

    query_results = graph.exec_query(
        "PREFIX voc: <https://swapi.co/vocabulary/> "
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        "SELECT ?eyeColor "
        "WHERE {"
        '  ?besalisk rdfs:label "Dexter Jettster" ; '
        "    voc:eyeColor ?eyeColor ."
        "}"
    )
    query_bindings = query_results.bindings  # List[Dict[str, Value]]
    res = [
        {k: v.value} for d in query_bindings for k, v in d.items()
    ]  # List[Dict[str, str]]

    assert len(res) == 1
    assert len(res[0]) == 1
    assert res[0]["eyeColor"] == "yellow"


def test_query_method_with_invalid_query() -> None:
    graph = OntotextGraphDBGraph(
        gdb_repository="http://localhost:7200/repositories/langchain",
    )
    from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

    with pytest.raises(QueryBadFormed) as e:
        graph.exec_query(
            "PREFIX : <https://swapi.co/vocabulary/> "
            "PREFIX owl: <http://www.w3.org/2002/07/owl#> "
            "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
            "PREFIX xsd: <http://www.w3.org/2001/XMLSchema#> "
            "SELECT ?character (MAX(?lifespan) AS ?maxLifespan) "
            "WHERE {"
            "  ?species a :Species ;"
            "    :character ?character ;"
            "    :averageLifespan ?lifespan ."
            "  FILTER(xsd:integer(?lifespan))"
            "} "
            "ORDER BY DESC(?maxLifespan) "
            "LIMIT 1"
        )
    assert str(e.value) == (
        "QueryBadFormed: A bad request has been sent to the endpoint: "
        "probably the SPARQL query is badly formed. \n\n"
        "Response:\n"
        "b\"MALFORMED QUERY: variable 'character' in projection "
        'not present in GROUP BY."'
    )
