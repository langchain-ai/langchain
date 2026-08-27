from langchain_classic.schema.document import __all__

EXPECTED_ALL = ["BaseDocumentTransformer", "Document"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
