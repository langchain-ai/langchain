from typing import Any
from unittest import mock

from langchain.vectorstores.nucliadb import NucliaDB


class attrdict(dict):
    def __getitem__(self, key: str) -> Any:
        value = dict.__getitem__(self, key)
        return attrdict(value) if isinstance(value, dict) else value

    __getattr__ = __getitem__


def FakeCreate(**args: Any) -> Any:
    def fn(self: Any, **kwargs: Any) -> str:
        return "fake_uuid"

    return fn


def FakeDelete(**args: Any) -> Any:
    def fn(self: Any, **kwargs: Any) -> None:
        return None

    return fn


def FakeFind(**args: Any) -> Any:
    def fn(self: Any, **kwargs: Any) -> Any:
        return attrdict(
            {
                "resources": {
                    "123": attrdict(
                        {
                            "fields": {
                                "456": attrdict(
                                    {
                                        "paragraphs": {
                                            "123/t/text/0-14": attrdict(
                                                {
                                                    "text": "This is a test",
                                                    "order": 0,
                                                }
                                            ),
                                        }
                                    }
                                )
                            },
                            "data": {
                                "texts": {
                                    "text": {
                                        "body": "This is a test",
                                    }
                                }
                            },
                            "extra": attrdict({"metadata": {"some": "metadata"}}),
                        }
                    )
                }
            }
        )

    return fn


def test_add_texts() -> None:
    with mock.patch(
        "nuclia.sdk.resource.NucliaResource.create",
        new_callable=FakeCreate,
    ):
        ndb = NucliaDB(knowledge_box="YOUR_KB_ID", local=False, api_key="YOUR_API_KEY")
        assert ndb.is_local is False
        ids = ndb.add_texts(["This is a new test", "This is a second test"])
        assert len(ids) == 2


def test_delete() -> None:
    with mock.patch(
        "nuclia.sdk.resource.NucliaResource.delete",
        new_callable=FakeDelete,
    ):
        ndb = NucliaDB(knowledge_box="YOUR_KB_ID", local=False, api_key="YOUR_API_KEY")
        success = ndb.delete(["123", "456"])
        assert success


def test_search() -> None:
    with mock.patch(
        "nuclia.sdk.search.NucliaSearch.find",
        new_callable=FakeFind,
    ):
        ndb = NucliaDB(knowledge_box="YOUR_KB_ID", local=False, api_key="YOUR_API_KEY")
        results = ndb.similarity_search("Who was inspired by Ada Lovelace?")
        assert len(results) == 1
        assert results[0].page_content == "This is a test"
        assert results[0].metadata["extra"]["some"] == "metadata"
        assert results[0].metadata["value"]["body"] == "This is a test"
