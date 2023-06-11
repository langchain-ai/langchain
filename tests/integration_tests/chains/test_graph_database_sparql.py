"""Test RDF/ SPARQL Graph Database Chain."""
from langchain.graphs import RdfGraph


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
