from pathlib import Path

from langchain.graphs import GraphDBGraph

"""
cd libs/langchain/tests/integration_tests/graphs/graphdb
./start.sh
"""


def test_get_schema_with_query() -> None:
    graph = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        query_ontology='CONSTRUCT {?s ?p ?o} FROM <https://swapi.co/ontology/> WHERE {?s ?p ?o}'
    )

    from rdflib import Graph
    assert len(Graph().parse(data=graph.get_schema, format='turtle')) == 843


def test_get_schema_from_file() -> None:
    graph_xml = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.xml')
    )
    graph_n3 = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.n3')
    )
    graph_ttl = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.ttl')
    )
    graph_nt = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.nt')
    )
    graph_trig = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.trig')
    )
    graph_nq = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.nq')
    )
    graph_jsonld = GraphDBGraph(
        query_endpoint='http://localhost:7200/repositories/langchain',
        local_file=str(Path(__file__).parent.parent / 'examples/starwars-ontology.jsonld')
    )

    from rdflib import Graph
    assert len(Graph().parse(data=graph_xml.get_schema, format='turtle')) == 843
    assert len(Graph().parse(data=graph_n3.get_schema, format='turtle')) == 843
    assert len(Graph().parse(data=graph_ttl.get_schema, format='turtle')) == 843
    assert len(Graph().parse(data=graph_nt.get_schema, format='turtle')) == 843
    assert len(Graph().parse(data=graph_trig.get_schema, format='turtle')) == 843
    assert len(Graph().parse(data=graph_nq.get_schema, format='turtle')) == 843
    assert len(Graph().parse(data=graph_jsonld.get_schema, format='turtle')) == 843
