from langchain_unstructured import __all__

EXPECTED_ALL = [
    "UnstructuredLoader",
    "__version__",
]


def test_all_imports() -> None:
    assert sorted(EXPECTED_ALL) == sorted(__all__)
