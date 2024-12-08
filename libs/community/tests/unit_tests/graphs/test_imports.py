from langchain_community.graphs import __all__, _module_lookup

EXPECTED_ALL = [
    "MemgraphGraph",
    "NetworkxEntityGraph",
    "Neo4jGraph",
    "NebulaGraph",
    "BaseNeptuneGraph",
    "NeptuneAnalyticsGraph",
    "NeptuneGraph",
    "NeptuneRdfGraph",
    "KuzuGraph",
    "HugeGraph",
    "RdfGraph",
    "ArangoGraph",
    "FalkorDBGraph",
    "TigerGraph",
    "OntotextGraphDBGraph",
    "OracleGraph",
    "GremlinGraph",
]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
    assert set(__all__) == set(_module_lookup.keys())
