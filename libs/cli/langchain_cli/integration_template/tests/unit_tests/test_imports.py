from __module_name__ import __all__

EXPECTED_ALL = [
    "__ModuleName__LLM",
    "Chat__ModuleName__",
    "__ModuleName__VectorStore",
    "__ModuleName__Embeddings",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
