"""Test RDF/ SPARQL Graph Database Chain."""
import os
import re

from langchain.chains.graph_qa.sparql import GraphSparqlQAChain
from langchain.graphs import RdfGraph
from langchain.llms.openai import OpenAI


def test_connect_file_rdf() -> None:
    """
    Test loading online resource.
    """
    berners_lee_card = "http://www.w3.org/People/Berners-Lee/card"

    graph = RdfGraph(
        source_file=berners_lee_card,
        standard="rdf",
    )

    query = """SELECT ?s ?p ?o\n""" """WHERE { ?s ?p ?o }"""

    output = graph.query(query)
    assert len(output) == 86


def test_sparql_select() -> None:
    """
    Test for generating and executing simple SPARQL SELECT query.
    """
    berners_lee_card = "http://www.w3.org/People/Berners-Lee/card"

    graph = RdfGraph(
        source_file=berners_lee_card,
        standard="rdf",
    )

    chain = GraphSparqlQAChain.from_llm(OpenAI(temperature=0), graph=graph)
    output = chain.run("What is Tim Berners-Lee's work homepage?")
    expected_output = (
        " The work homepage of Tim Berners-Lee is "
        "http://www.w3.org/People/Berners-Lee/."
    )
    assert output == expected_output


def test_sparql_insert() -> None:
    """
    Test for generating and executing simple SPARQL INSERT query.
    """
    berners_lee_card = "http://www.w3.org/People/Berners-Lee/card"
    _local_copy = "test.ttl"

    graph = RdfGraph(
        source_file=berners_lee_card,
        standard="rdf",
        local_copy=_local_copy,
    )

    chain = GraphSparqlQAChain.from_llm(OpenAI(temperature=0), graph=graph)
    chain.run(
        "Save that the person with the name 'Timothy Berners-Lee' "
        "has a work homepage at 'http://www.w3.org/foo/bar/'"
    )
    query = (
        """PREFIX foaf: <http://xmlns.com/foaf/0.1/>\n"""
        """SELECT ?hp\n"""
        """WHERE {\n"""
        """    ?person foaf:name "Timothy Berners-Lee" . \n"""
        """    ?person foaf:workplaceHomepage ?hp .\n"""
        """}"""
    )
    output = graph.query(query)
    assert len(output) == 2

    # clean up
    try:
        os.remove(_local_copy)
    except OSError:
        pass


def test_loading_schema_from_graphdb() -> None:
    graph = RdfGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        graph_kwargs={"bind_namespaces": "none"},
    )
    schema = graph.get_schema
    prefix = (
        "In the following, each IRI is followed by the local name and "
        "optionally its description in parentheses. \n"
        "The RDF graph supports the following node types:"
    )
    assert schema.startswith(prefix)

    infix = "The RDF graph supports the following relationships:"
    assert infix in schema

    classes = schema[len(prefix) : schema.index(infix)]
    assert len(re.findall("<[^>]+> \\([^)]+\\)", classes)) == 5

    relationships = schema[schema.index(infix) + len(infix) :]
    assert len(re.findall("<[^>]+> \\([^)]+\\)", relationships)) == 58


def test_graph_qa_chain_with_graphdb() -> None:
    graph = RdfGraph(
        query_endpoint="http://localhost:7200/repositories/langchain",
        graph_kwargs={"bind_namespaces": "none"},
    )

    chain = GraphSparqlQAChain.from_llm(OpenAI(temperature=0), graph=graph)
    output = chain.run("What is Tim Berners-Lee's work homepage?")
    expected_output = (
        " The work homepage of Tim Berners-Lee is "
        "http://www.w3.org/People/Berners-Lee/."
    )
    assert output == expected_output
