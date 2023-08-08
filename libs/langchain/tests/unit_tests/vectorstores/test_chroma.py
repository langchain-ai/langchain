import pytest

from langchain.vectorstores.chroma import _filter_list_metadata


def test_filter_list_metadata(caplog: pytest.LogCaptureFixture) -> None:
    metadata = [
        {"key1": "this is a string!", "key2": ["a", "list", "of", "strings"]},
        {
            "key1": "this is another string!",
            "key2": ["another", "list", "of", "strings"],
        },
    ]

    filtered_metadata = _filter_list_metadata(metadata)
    assert filtered_metadata == [
        {"key1": "this is a string!"},
        {"key1": "this is another string!"},
    ]
    assert "List metadata detected. Skipping." in caplog.text
