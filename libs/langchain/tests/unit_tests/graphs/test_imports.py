from langchain import graphs
from tests.unit_tests import assert_all_importable

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
    assert_all_importable(graphs)
