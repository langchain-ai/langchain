from pathlib import Path

import pytest

from langchain_community.graphs import OntotextGraphDBGraph

"""
cd libs/community/tests/integration_tests/graphs/docker-compose-ontotext-graphdb
./start.sh
"""


def test_query_method_with_valid_query() -> None:
    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        query_ontology="CONSTRUCT {?s ?p ?o}"
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )

    query_results = graph.query(
        "PREFIX voc: <https://swapi.co/vocabulary/> "
        "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
        "SELECT ?eyeColor "
        "WHERE {"
        '  ?besalisk rdfs:label "Dexter Jettster" ; '
        "    voc:eyeColor ?eyeColor ."
        "}"
    )
    query_bindings = query_results.bindings  # type: ignore[union-attr]
    res = [{k: v.value} for d in query_bindings for k, v in d.items()]

    assert len(res) == 1
    assert len(res[0]) == 1
    assert res[0]["eyeColor"] == "yellow"


def test_query_method_with_invalid_query() -> None:
    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        query_ontology="CONSTRUCT {?s ?p ?o}"
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )

    from SPARQLWrapper.SPARQLExceptions import QueryBadFormed

    with pytest.raises(QueryBadFormed) as e:
        graph.query(
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


def test_get_schema_with_query() -> None:
    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        query_ontology="CONSTRUCT {?s ?p ?o}"
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )

    from rdflib import Graph

    assert len(Graph().parse(data=graph.get_schema, format="turtle")) == 19


@pytest.mark.parametrize(
    "rdf_format, file_extension",
    [
        ("json-ld", "json"),
        ("json-ld", "jsonld"),
        ("json-ld", "json-ld"),
        ("xml", "rdf"),
        ("xml", "xml"),
        ("xml", "owl"),
        ("pretty-xml", "xml"),
        ("n3", "n3"),
        ("turtle", "ttl"),
        ("nt", "nt"),
        ("trig", "trig"),
        ("nquads", "nq"),
        ("nquads", "nquads"),
        ("trix", "trix"),
    ],
)
def test_get_schema_from_file(
    tmp_path: Path, rdf_format: str, file_extension: str
) -> None:
    expected_number_of_ontology_statements = 19

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        query_ontology="CONSTRUCT {?s ?p ?o}"
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )

    from rdflib import ConjunctiveGraph, Graph

    assert (
        len(Graph().parse(data=graph.get_schema, format="turtle"))
        == expected_number_of_ontology_statements
    )

    # serialize the ontology schema loaded with the query in a local file
    # in various rdf formats and check that this results
    # in the same number of statements
    conjunctive_graph = ConjunctiveGraph()
    ontology_context = conjunctive_graph.get_context("https://swapi.co/ontology/")
    ontology_context.parse(data=graph.get_schema, format="turtle")

    assert len(ontology_context) == expected_number_of_ontology_statements
    assert len(conjunctive_graph) == expected_number_of_ontology_statements

    local_file = tmp_path / ("starwars-ontology." + file_extension)
    conjunctive_graph.serialize(local_file, format=rdf_format)

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        local_file=str(local_file),
    )
    assert (
        len(Graph().parse(data=graph.get_schema, format="turtle"))
        == expected_number_of_ontology_statements
    )


@pytest.mark.parametrize(
    "rdf_format", ["json-ld", "xml", "n3", "turtle", "nt", "trig", "nquads", "trix"]
)
def test_get_schema_from_file_with_explicit_rdf_format(
    tmp_path: Path, rdf_format: str
) -> None:
    expected_number_of_ontology_statements = 19

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        query_ontology="CONSTRUCT {?s ?p ?o}"
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )

    from rdflib import ConjunctiveGraph, Graph

    assert (
        len(Graph().parse(data=graph.get_schema, format="turtle"))
        == expected_number_of_ontology_statements
    )

    # serialize the ontology schema loaded with the query in a local file
    # in various rdf formats and check that this results
    # in the same number of statements
    conjunctive_graph = ConjunctiveGraph()
    ontology_context = conjunctive_graph.get_context("https://swapi.co/ontology/")
    ontology_context.parse(data=graph.get_schema, format="turtle")

    assert len(ontology_context) == expected_number_of_ontology_statements
    assert len(conjunctive_graph) == expected_number_of_ontology_statements

    local_file = tmp_path / "starwars-ontology.txt"
    conjunctive_graph.serialize(local_file, format=rdf_format)

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        local_file=str(local_file),
        local_file_format=rdf_format,
    )
    assert (
        len(Graph().parse(data=graph.get_schema, format="turtle"))
        == expected_number_of_ontology_statements
    )


def test_get_schema_from_file_with_wrong_extension(tmp_path: Path) -> None:
    expected_number_of_ontology_statements = 19

    graph = OntotextGraphDBGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        query_ontology="CONSTRUCT {?s ?p ?o}"
        "FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}",
    )

    from rdflib import ConjunctiveGraph, Graph

    assert (
        len(Graph().parse(data=graph.get_schema, format="turtle"))
        == expected_number_of_ontology_statements
    )

    conjunctive_graph = ConjunctiveGraph()
    ontology_context = conjunctive_graph.get_context("https://swapi.co/ontology/")
    ontology_context.parse(data=graph.get_schema, format="turtle")

    assert len(ontology_context) == expected_number_of_ontology_statements
    assert len(conjunctive_graph) == expected_number_of_ontology_statements

    local_file = tmp_path / "starwars-ontology.trig"
    conjunctive_graph.serialize(local_file, format="nquads")

    with pytest.raises(ValueError):
        OntotextGraphDBGraph(
            query_endpoint="http://localhost:7200/repositories/langchain",
            local_file=str(local_file),
        )
