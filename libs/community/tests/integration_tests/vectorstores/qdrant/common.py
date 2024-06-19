from typing import List

from langchain_core.documents import Document


def qdrant_is_not_running() -> bool:
    """Check if Qdrant is not running."""
    import requests

    try:
        response = requests.get("http://localhost:6333", timeout=10.0)
        response_json = response.json()
        return response_json.get("title") != "qdrant - vector search engine"
    except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
        return True


def assert_documents_equals(actual: List[Document], expected: List[Document]):  # type: ignore[no-untyped-def]
    assert len(actual) == len(expected)

    for actual_doc, expected_doc in zip(actual, expected):
        assert actual_doc.page_content == expected_doc.page_content

        assert "_id" in actual_doc.metadata
        assert "_collection_name" in actual_doc.metadata

        actual_doc.metadata.pop("_id")
        actual_doc.metadata.pop("_collection_name")

        assert actual_doc.metadata == expected_doc.metadata
