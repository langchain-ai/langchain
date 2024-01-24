from datetime import datetime
from typing import List

import pytest
from langchain_core.embeddings import Embeddings

from langchain_community.indexes import MemoryRecordManager


class _FakeEmbeddings(Embeddings):
    """Fake embeddings functionality for testing."""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Return simple embeddings.
        Embeddings encode each text as its index."""
        return [[float(1.0)] * 9 + [float(i)] for i in range(len(texts))]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """Return constant query embeddings.
        Embeddings are identical to embed_documents(texts)[0].
        Distance to each text will be that text's index,
        as it was passed to embed_documents."""
        return [float(1.0)] * 9 + [float(0.0)]

    async def aembed_query(self, text: str) -> List[float]:
        return self.embed_query(text)


# def test_try_update() -> None:
#     """
#         Args:
#             keys: A list of record keys to upsert.
#             group_ids: A list of group IDs corresponding to the keys.
#             time_at_least: if provided, updates should only happen if the
#               updated_at field is at least this time.
#     """
#     with tempfile.NamedTemporaryFile(suffix="-sqlite.db") as filename:
#         # record_manager = SQLRecordManager(
#         #     namespace="record_manager_cache",
#         #     db_url=f"sqlite:///{filename.name}"
#         # )
#         record_manager = MemoryRecordManager(
#             namespace="record_manager_cache"
#         )
#         documents = [
#             Document(page_content="hello",
#                      metadata={"id": 1}), ]
#         record_manager.create_schema()
#         vector_store = FAISS.from_documents(
#             documents=documents,
#             embedding=_FakeEmbeddings()
#         )
#         index_kwargs = {
#             "record_manager": record_manager,
#             "vector_store": vector_store,
#             "source_id_key": "id",
#         }
#         result = index(
#             docs_source=documents,
#             cleanup="incremental",
#             **index_kwargs)
#         assert result["num_added"] == 1
#         assert result["num_deleted"] == 0
#         documents[0].page_content = "modif"
#         result = index(
#             docs_source=documents,
#             cleanup="full",
#             **index_kwargs)
#         assert result["num_added"] == 1
#         assert result["num_deleted"] == 1


def test_update_exist_list_keys() -> None:
    record_manager = MemoryRecordManager(namespace="record_manager_cache")
    keys = [
        "key1",
    ]
    group_ids = ["1"]
    time_at_least = datetime.now().timestamp()
    record_manager.create_schema()

    record_manager.get_time()

    record_manager.update(group_ids=group_ids, keys=keys, time_at_least=time_at_least)
    assert record_manager.exists(keys)
    assert record_manager.exists(["a", "key1", "c"]) == [False, True, False]
    assert len(record_manager.list_keys()) == 1
    assert (
        len(
            record_manager.list_keys(
                before=time_at_least + 1000.0,
                after=time_at_least - 1000.0,
                group_ids=group_ids,
            )
        )
        == 1
    )

    assert (
        len(
            record_manager.list_keys(
                before=time_at_least + 1000.0,
            )
        )
        == 1
    )
    assert (
        len(
            record_manager.list_keys(
                after=time_at_least - 1000.0,
            )
        )
        == 1
    )
    assert len(record_manager.list_keys(group_ids=group_ids)) == 1
    assert len(record_manager.list_keys(group_ids=["2"])) == 0
    assert (
        len(
            record_manager.list_keys(
                before=time_at_least - 1000.0,
            )
        )
        == 0
    )
    assert (
        len(
            record_manager.list_keys(
                after=time_at_least + 1000.0,
            )
        )
        == 0
    )

    record_manager.delete_keys(keys=keys)
    assert len(record_manager.list_keys()) == 0


@pytest.mark.asyncio
async def test_aupdate_exist_list_keys() -> None:
    record_manager = MemoryRecordManager(namespace="record_manager_cache")
    keys = [
        "key1",
    ]
    group_ids = ["1"]
    time_at_least = datetime.now().timestamp()
    record_manager.create_schema()

    await record_manager.aget_time()

    await record_manager.aupdate(
        group_ids=group_ids, keys=keys, time_at_least=time_at_least
    )
    assert await record_manager.aexists(keys)
    assert len(await record_manager.alist_keys()) == 1
    assert (
        len(
            await record_manager.alist_keys(
                before=time_at_least + 1000.0,
                after=time_at_least - 1000.0,
                group_ids=group_ids,
            )
        )
        == 1
    )

    assert (
        len(
            await record_manager.alist_keys(
                before=time_at_least + 1000.0,
            )
        )
        == 1
    )
    assert (
        len(
            await record_manager.alist_keys(
                after=time_at_least - 1000.0,
            )
        )
        == 1
    )
    assert len(await record_manager.alist_keys(group_ids=group_ids)) == 1
    assert len(await record_manager.alist_keys(group_ids=["2"])) == 0
    assert (
        len(
            await record_manager.alist_keys(
                before=time_at_least - 1000.0,
            )
        )
        == 0
    )
    assert (
        len(
            await record_manager.alist_keys(
                after=time_at_least + 1000.0,
            )
        )
        == 0
    )

    await record_manager.adelete_keys(keys=keys)
    assert len(await record_manager.alist_keys()) == 0
