from langchain_core.documents import Document


def test_lc_namespace() -> None:
    assert Document.get_lc_namespace() == [
        "langchain",
        "schema",
        "document",
    ]
