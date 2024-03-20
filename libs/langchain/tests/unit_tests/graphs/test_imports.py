import pytest

from langchain import graphs

EXPECTED_DEPRECATED_IMPORTS = [
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


def test_deprecated_imports() -> None:
    for import_ in EXPECTED_DEPRECATED_IMPORTS:
        with pytest.raises(ImportError) as e:
            getattr(graphs, import_)
            assert "langchain_community" in e
    with pytest.raises(AttributeError):
        getattr(graphs, "foo")
