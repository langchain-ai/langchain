"""Test Cloudflare Vectorize.

In order to run this test, you need to:
1. Have a Cloudflare account
2. Set up API tokens with access to:
   - Workers AI
   - Vectorize
   - D1 (optional, for raw value storage)
3. Set environment variables in .env file:
   CF_ACCOUNT_ID
   CF_API_TOKEN or (CF_VECTORIZE_TOKEN, CF_D1_TOKEN)
   CF_D1_DATABASE_ID (optional)
"""

import os
import uuid
from typing import List

import pytest
from langchain_core.documents import Document

from langchain_community.embeddings.cloudflare_workersai import CloudflareWorkersAIEmbeddings
from langchain_community.vectorstores.cloudflare_vectorize import CloudflareVectorize

MODEL_WORKERSAI = "@cf/baai/bge-large-en-v1.5"

@pytest.fixture(scope="class")
def embeddings() -> CloudflareWorkersAIEmbeddings:
    """Get embeddings model."""
    return CloudflareWorkersAIEmbeddings(
        account_id=os.getenv("CF_ACCOUNT_ID"),
        api_token=os.getenv("CF_AI_TOKEN"),
        model_name=MODEL_WORKERSAI,
    )

@pytest.fixture(scope="class")
def store(embeddings):
    index_name = f"test-langchain-{uuid.uuid4().hex}"
    
    store = CloudflareVectorize(
        embedding=embeddings,
        account_id=os.getenv("CF_ACCOUNT_ID"),
        d1_database_id=os.getenv("CF_D1_DATABASE_ID"),
        vectorize_api_token=os.getenv("CF_VECTORIZE_TOKEN"),
        d1_api_token=os.getenv("CF_D1_TOKEN"),
        index_name=index_name
    )
    
    # Create the index
    store.create_index(wait=True)
    store.create_metadata_index(property_name="section", index_type="string", wait=True)
    store.add_documents(documents=TestCloudflareVectorize.documents, wait=True)
    
    yield store
    
    # Cleanup
    store.delete_index()


class TestCloudflareVectorize:
    """Test Cloudflare Vectorize functionality."""

    index_name = f"test-langchain-{uuid.uuid4().hex}"
    documents: List[Document] = [
        Document(
            page_content="Cloudflare's headquarters are in San Francisco, California.",
            metadata={"section": "Introduction"}
        ),
        Document(
            page_content="Cloudflare launched Workers AI, an AI inference platform.",
            metadata={"section": "Products"}
        ),
        Document(
            page_content="Cloudflare provides edge computing and CDN services.",
            metadata={"section": "Products"}
        ),
        Document(
            page_content="Cloudflare offers SASE and Zero Trust solutions.",
            metadata={"section": "Security"}
        ),
    ]

    def test_similarity_search(self, store: CloudflareVectorize) -> None:
        """Test similarity search."""
        docs = store.similarity_search(
            query="AI platform",
            k=2
        )
        assert len(docs) > 0
        assert any("Workers AI" in doc.page_content for doc in docs)

    def test_similarity_search_with_score(self, store: CloudflareVectorize) -> None:
        """Test similarity search with scores."""
        docs, scores = store.similarity_search_with_score(
            query="AI platform",
            k=2
        )
        assert len(docs) > 0
        assert len(scores) > 0
        assert all(isinstance(score, float) for score in scores)
        assert any("Workers AI" in doc.page_content for doc in docs)

    def test_similarity_search_with_metadata_filter(self, store: CloudflareVectorize) -> None:
        """Test similarity search with metadata filtering."""
        docs = store.similarity_search(
            query="Cloudflare services",
            k=2,
            md_filter={"section": "Products"},
            return_metadata='all'
        )
        assert len(docs) > 0
        assert all(doc.metadata["section"] == "Products" for doc in docs)

    def test_get_by_ids(self, store: CloudflareVectorize) -> None:
        """Test retrieving documents by IDs."""
        # First get some IDs via search
        docs = store.similarity_search(
            query="California",
            k=1
        )
        doc_ids = set(doc.id for doc in docs)
        
        retrieved_docs = store.get_by_ids(
            ids=list(doc_ids)
        )
        retrieved_ids = set(doc.id for doc in retrieved_docs)
        
        assert retrieved_ids == doc_ids, f"Retrieved IDs {retrieved_ids} don't match original IDs {doc_ids}"
        
    def test_add_duplicates_and_upsert(self, store: CloudflareVectorize):
        """Test adding duplicate documents and upserting documents."""
        
        # Initial documents
        docs = [
            Document(
                id="test-id-1",
                page_content="This is a test document",
                metadata={"section": "Introduction"}
            ),
            Document(
                id="test-id-2", 
                page_content="Another test document",
                metadata={"section": "Introduction"}
            )
        ]
        
        # Add initial documents
        store.add_documents(documents=docs, wait=True)
        
        # Try adding same documents again (should not create duplicates)
        store.add_documents(documents=docs, wait=True)
        
        # Search to verify no duplicates        
        results = store.get_by_ids(ids=["test-id-1", "test-id-2"])
        
        assert len(results) == 2, "Should only have 2 documents despite adding twice"
        
        # Update document content
        updated_doc = Document(
            id="test-id-1",
            page_content="Updated: This is a test document",
            metadata={"section": "Introduction"}
        )
        
        # Upsert the updated document
        store.add_documents(
            documents=[updated_doc],
            upsert=True,
            wait=True
        )
        
        # Verify update
        results = store.get_by_ids(ids=["test-id-1"])
        assert len(results) == 1
        assert results[0].page_content.startswith("Updated:")

    def test_delete_and_verify(self, store: CloudflareVectorize):
        """Test deleting documents and verifying they're gone."""
        test_id = uuid.uuid4().hex[:8]
        
        # Initial documents with unique IDs
        docs = [
            Document(
                id=f"{test_id}-delete-test-1",
                page_content="Document to delete 1",
                metadata={"section": "Test"}
            ),
            Document(
                id=f"{test_id}-delete-test-2",
                page_content="Document to delete 2",
                metadata={"section": "Test"}
            ),
            Document(
                id=f"{test_id}-keep-test-1",
                page_content="Document to keep",
                metadata={"section": "Test"}
            )
        ]
        
        # Add documents
        store.add_documents(documents=docs, wait=True)
        
        # Verify initial state
        results = store.get_by_ids(ids=[f"{test_id}-delete-test-1", f"{test_id}-delete-test-2", f"{test_id}-keep-test-1"])
        assert len(results) == 3, "Should have all 3 documents initially"
        
        # Delete specific documents
        ids_to_delete = [f"{test_id}-delete-test-1", f"{test_id}-delete-test-2"]
        store.delete(ids=ids_to_delete, wait=True)
        
        # Verify deletion
        results = store.get_by_ids(ids=ids_to_delete)
        assert len(results) == 0
        
        # Verify remaining document
        results = store.get_by_ids(ids=[f"{test_id}-keep-test-1"])
        assert len(results) == 1
        assert results[0].id == f"{test_id}-keep-test-1"


    def test_similarity_search_with_namespace(self, store: CloudflareVectorize):
        """Test similarity search with namespace filtering."""
        # Create unique namespace for this test
        test_namespace = f"test-namespace-{uuid.uuid4().hex[:8]}"
        
        # Documents to add with the namespace
        namespace_docs = [
            Document(
                page_content="Cloudflare R2 provides S3-compatible object storage.",
                metadata={"section": "Products"}
            ),
            Document(
                page_content="Cloudflare Pages is a JAMstack platform for frontend developers.",
                metadata={"section": "Products"}
            )
        ]
        
        # Add documents with namespace
        store.add_documents(
            documents=namespace_docs, 
            namespaces=[test_namespace] * len(namespace_docs),
            wait=True
        )
        
        # Search within the namespace
        results = store.similarity_search(
            query="storage solution",
            k=2,
            namespace=test_namespace
        )
        
        # Verify results
        assert len(results) > 0
        assert any("R2" in doc.page_content for doc in results)
        
        # Verify namespace filtering works by searching with a different namespace
        other_namespace = f"nonexistent-namespace-{uuid.uuid4().hex[:8]}"
        empty_results = store.similarity_search(
            query="storage solution",
            k=2,
            namespace=other_namespace
        )
        
        # Should find no results in the other namespace
        assert len(empty_results) == 0
        

    def test_from_documents(self, embeddings: CloudflareWorkersAIEmbeddings) -> None:
        """Test creating store from documents."""
        new_index = f"test-langchain-{uuid.uuid4().hex}"
        try:
            store = CloudflareVectorize.from_documents(
                documents=self.documents,
                embedding=embeddings,
                account_id=os.getenv("CF_ACCOUNT_ID"),
                index_name=new_index,
                d1_database_id=os.getenv("CF_D1_DATABASE_ID"),
                vectorize_api_token=os.getenv("CF_VECTORIZE_TOKEN"),
                d1_api_token=os.getenv("CF_D1_TOKEN"),
                wait=True
            )
            
            docs = store.similarity_search(
                query="California",
                k=1,
                index_name=new_index
            )
            assert len(docs) > 0
            assert "California" in docs[0].page_content
            
        finally:
            # Cleanup
            store.delete_index(new_index)