import tempfile
import uuid
from typing import Optional

import pytest
# from langchain_core.documents import Document

from langchain_community.vectorstores import Qdrant
from langchain_community.vectorstores.qdrant import QdrantException
from tests.integration_tests.vectorstores.fake_embeddings import (
    ConsistentFakeEmbeddings,
)

# test_qdrant_from_existing_client_reuse_same_client
@pytest.mark.parametrize("collection_name", ["custom-collection"])
def test_qdrant_from_existing_index_uses_same_collection(collection_name: Optional[str]) -> None:
    """Test if the Qdrant.from_existing_client reuses the some client."""
    from qdrant_client import QdrantClient
    
    # embeddings = ConsistentFakeEmbeddings()
    with tempfile.TemporaryDirectory() as tmpdir:
        
        docs = ["foo"]
        qdrant = Qdrant.from_texts(docs, embedding=ConsistentFakeEmbeddings(), path=str(tmpdir), collection_name=collection_name, )
        del qdrant  
        
        
        qdrant = Qdrant.from_existing_index(embedding=ConsistentFakeEmbeddings(), path=str(tmpdir), collection_name=collection_name)
        qdrant.add_texts(["baz","bar"])
        del qdrant
        
        client = QdrantClient(path=str(tmpdir))
        assert 3 == client.count(collection_name).count
        
        
        
        
        
        
        
    
    


