from langchain_classic import graphs

EXPECTED_ALL = [
    "MemgraphGraph",
    "NetworkxEntityGraph",
    "Neo4jGraph",
    "NebulaGraph",
    "NeptuneGraph",
    "KuzuGraph",
    "HugeGraph",
    "RdfGraph",
    "ArangoGraph",
    "FalkorDBGraph",
]


def test_all_imports() -> None:
    assert set(graphs.__all__) == set(EXPECTED_ALL)
