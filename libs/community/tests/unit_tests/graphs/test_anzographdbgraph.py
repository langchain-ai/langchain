@pytest.mark.requires(rdflib)
@pytest.mark.requires(sparqlwrapper)
def test_import() -> None:
    from langchain_community.graphs.anzograph_graph import (
        AnzoGraphDBGraph,
    )