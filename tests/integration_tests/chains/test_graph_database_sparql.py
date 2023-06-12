"""Test RDF/ SPARQL Graph Database Chain."""
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


def test_sparql_generating_run() -> None:
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
    print(output)
    expected_output = " The work homepage of Tim Berners-Lee is http://www.w3.org/People/Berners-Lee/."
    assert output == expected_output
