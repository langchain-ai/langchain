from typing import Optional, Dict, Any, List
from unittest.mock import patch, Mock, AsyncMock

import numpy
import pytest

from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.supabase import AsyncSupabaseVectorStore


async def create_vector_store() -> AsyncSupabaseVectorStore:
    from supabase.client import acreate_client
    import os
    return AsyncSupabaseVectorStore(
        client=await acreate_client(os.environ["my_supabase_url"], os.environ["my_supabase_key"]),
        embedding=FakeEmbeddings(size=3),
        table_name="documents",
        #query_name="match_documents",
    )

@pytest.mark.requires("supabase")
async def test_ids_used_correctly() -> None:
    """Check whether vector store uses the document ids when provided with them."""
    from langchain_core.documents import Document

    documents = [
        Document(id="id1",
            page_content="page zero Lorem Ipsum",
            metadata={"source": "document.pdf", "page": 0, "id": "ID-document-1"},
        ),
        Document(
            id="id2",
            page_content="page one Lorem Ipsum Dolor sit ameit",
            metadata={"source": "document.pdf", "page": 1, "id": "ID-document-2"},
        ),
    ]
    ids_provided = [i.id for i in documents]
    table_mock = Mock(name="from_()")
    mock_upsert = AsyncMock()
    table_mock.upsert.return_value = mock_upsert
    mock_result = Mock()
    mock_upsert.execute.return_value = mock_result
    mock_result.data=[{"id":"id1"}, {"id": "id2"}]

    import supabase
    with patch.object(
         supabase._async.client.AsyncClient, "from_",return_value=table_mock) as from_mock:
    # ), patch.object(supabase._async.client.AsyncClient, "get_index", mock_default_index):
    #:
        vector_store = await create_vector_store()
        ids_used_at_upload = await vector_store.aadd_documents(documents, ids=ids_provided)
        assert len(ids_provided) == len(ids_used_at_upload)
        assert ids_provided == ids_used_at_upload
        from_mock.assert_called_once_with("documents")
    table_mock.upsert.assert_called_once()
    list_submitted = table_mock.upsert.call_args.args[0]
    assert len(list_submitted) == 2
    assert [d["id"] for d in list_submitted] == [d.id for d in  documents]
    assert [d["content"] for d in list_submitted] == [d.page_content for d in documents]
    for obj in list_submitted:
        assert len(obj["embedding"])==3
        assert all(type(v)==type(numpy.float64(0.2)) for v in obj["embedding"])
    assert [d["metadata"] for d in list_submitted] == [d.metadata for d in documents]


