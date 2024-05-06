from langchain_core.documents import __all__

EXPECTED_ALL = ["Document", "BaseDocumentTransformer", "BaseDocumentCompressor"]


def test_all_imports() -> None:
    assert set(__all__) == set(EXPECTED_ALL)
