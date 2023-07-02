"""Test RDF/ SPARQL Graph Database Chain."""
import os
import pytest

from langchain.chains.graph_qa.sparql import GraphSparqlQAChain
from langchain.graphs import RdfGraph
from langchain.llms.openai import OpenAI


def test_connect_file_rdf() -> None:
    """
    Test loading online resource.
    """
    url = "http://www.w3.org/People/Berners-Lee/card"

    graph = RdfGraph(
        url=url,
        standard="rdf",
    )

    query = (
        """SELECT ?s ?p ?o\n"""
        """WHERE { ?s ?p ?o }"""
    )

    output = graph.query(query)
    assert len(output) == 86

# TODO: test for RDFS and OWL, too


def test_sparql_select() -> None:
    """
    Test for generating and executing simple SPARQL SELECT query.
    """
    url = "http://www.w3.org/People/Berners-Lee/card"

    graph = RdfGraph(
        url=url,
        standard="rdf",
    )

    chain = GraphSparqlQAChain.from_llm(OpenAI(temperature=0), graph=graph)
    output = chain.run("What is Tim Berners-Lee's work homepage?")
    expected_output = " The work homepage of Tim Berners-Lee is http://www.w3.org/People/Berners-Lee/."
    assert output == expected_output


def test_sparql_insert() -> None:
    """
    Test for generating and executing simple SPARQL INSERT query.
    """
    url = "http://www.w3.org/People/Berners-Lee/card"
    _local_file = "test.ttl"

    graph = RdfGraph(
        url=url,
        standard="rdf",
        local_file=_local_file,
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
        os.remove(_local_file)
    except OSError:
        pass
