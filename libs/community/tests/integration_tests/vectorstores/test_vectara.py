import os

import pytest
import requests
from langchain_core.documents import Document
from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores.vectara import (
    VectaraQueryConfig,
    SearchConfig,
    GenerationConfig,
    CorpusConfig,
    MmrReranker,
    File,
    TableExtractionConfig,
    ChunkingStrategy
)

#
# For this test to run properly, please setup as follows:
# 1. Create a Vectara account: sign up at https://www.vectara.com/integrations/langchain
# 2. Create a corpus in your Vectara account
# 3. Create an API_KEY for this corpus with permissions for query and indexing
# 4. Setup environment variables:
#    VECTARA_API_KEY and VECTARA_CORPUS_key
#

test_prompt_name = "vectara-summary-ext-24-05-med-omni"

@pytest.fixture(scope="module")
def vectara():
    api_key = os.getenv("VECTARA_API_KEY")
    if not api_key:
        pytest.skip("VECTARA_API_KEY environment variable not set")
    vectara_instance = Vectara(vectara_api_key=api_key)

    yield vectara_instance

    cleanup_documents(vectara_instance, os.getenv("VECTARA_CORPUS_KEY"))
def cleanup_documents(vectara, corpus_key):
    """
    Fetch all documents from the corpus and delete them after tests are completed.
    """

    url = f"https://api.vectara.io/v2/corpora/{corpus_key}/documents"
    headers = {
        'Accept': 'application/json',
        'x-api-key': vectara._vectara_api_key
    }

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return

    data = response.json()
    document_ids = [doc["id"] for doc in data.get("documents", [])]

    if not document_ids:
        return

    vectara.delete(corpus_key, ids=document_ids)
@pytest.fixture(scope="module")
def corpus_key():
    corpus_key = os.getenv("VECTARA_CORPUS_KEY")
    if not corpus_key:
        pytest.skip("VECTARA_CORPUS_KEY environment variable not set")
    return corpus_key

@pytest.fixture(scope="module")
def add_test_doc(vectara, corpus_key):
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "The lazy dog sleeps while the quick brown fox jumps.",
        "A fox is quick and brown in color."
    ]
    vectara.add_texts(texts=texts, corpus_key=corpus_key)
def test_initialization(vectara):
    """Check that the instance is initialized properly."""
    api_key = os.getenv("VECTARA_API_KEY")
    assert vectara._vectara_api_key == api_key

def test_get_post_headers(vectara):
    """Ensure headers are correctly formed."""
    headers = vectara._get_post_headers()
    assert "x-api-key" in headers
    assert "Content-Type" in headers
def test_add_texts(vectara, corpus_key):
    """Index texts and verify they are indexed successfully."""
    texts = [
        "This is a test document of type core.",
        "This is another test document.",
        "This is a third test document."
    ]
    metadatas = [
        {"source": "test1"},
        {"source": "test2"},
        {"source": "test3"}
    ]

    # Test adding texts with core document type
    doc_ids = vectara.add_texts(
        texts=texts,
        metadatas=metadatas,
        doc_type="core",
        corpus_key=corpus_key
    )
    assert len(doc_ids) == 1
    assert isinstance(doc_ids[0], str)

    # Test adding texts with structured document type
    texts = [
        "This is a test document of type structure.",
        "This is another test document.",
        "This is a third test document."
    ]
    metadatas = [
        {"source": "test1"},
        {"source": "test2"},
        {"source": "test3"}
    ]

    doc_ids = vectara.add_texts(
        texts=texts,
        metadatas=metadatas,
        doc_type="structured",
        corpus_key=corpus_key
    )
    assert len(doc_ids) == 1
    assert isinstance(doc_ids[0], str)

def test_add_files(vectara, corpus_key, tmp_path):
    """Test uploading a file for indexing."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("This is a test file content.")

    file_obj = File(
        file_path=str(test_file),
        metadata={"source": "test_file"},
        chunking_strategy=ChunkingStrategy(max_chars_per_chunk=150),
    )

    doc_ids = vectara.add_files([file_obj], corpus_key)
    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)

def test_similarity_search(vectara, corpus_key, add_test_doc):
    """Run a similarity search query."""
    # Test basic similarity search
    results = vectara.similarity_search(
        query="What color is the fox?",
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        )
    )
    assert len(results) > 0
    assert isinstance(results[0], Document)

    # Test similarity search with scores
    results_with_scores = vectara.similarity_search_with_score(
        query="What color is the fox?",
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        )
    )
    assert len(results_with_scores) > 0
    assert isinstance(results_with_scores[0], tuple)
    assert isinstance(results_with_scores[0][0], Document)
    assert isinstance(results_with_scores[0][1], float)

def test_mmr_search(vectara, corpus_key, add_test_doc):
    query = "What color is the fox?"
    results = vectara.max_marginal_relevance_search(
        query=query,
        fetch_k=5,
        lambda_mult=0.5,
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        )
    )
    assert len(results) > 0
    assert isinstance(results[0], Document)

def test_delete_documents(vectara, corpus_key):
    texts = ["This is a test document to be deleted."]
    doc_ids = vectara.add_texts(
        texts=texts,
        corpus_key=corpus_key
    )

    success = vectara.delete(corpus_key=corpus_key, ids=doc_ids)
    assert success is True

def test_vectara_as_rag(vectara, corpus_key, add_test_doc):
    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        ),
        generation=GenerationConfig(
            max_results=5,
            response_lang="eng"
        )
    )

    rag = vectara.as_rag(config)

    result = rag.invoke("What color is the fox?")
    assert "question" in result
    assert "answer" in result
    assert "context" in result

def test_streaming(vectara, corpus_key, add_test_doc):
    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        ),
        generation=GenerationConfig(
            max_results=5
        ),
        stream_response=True
    )

    rag = vectara.as_rag(config)
    chunks = list(rag.stream("What color is the fox?"))

    assert len(chunks) > 0
    assert any("question" in chunk for chunk in chunks)
    assert any("answer" in chunk for chunk in chunks)
    assert any("context" in chunk for chunk in chunks)

def test_as_chat(vectara, corpus_key, add_test_doc):
    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        ),
        chat=True
    )

    chat = vectara.as_chat(config)
    result = chat.invoke("Tell me about the fox")

    assert "question" in result
    assert "answer" in result
    assert "chat_id" in result


def test_vectara_query(vectara, corpus_key):
    """Test vectara_query with and without streaming."""

    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        ),
        generation=GenerationConfig(
            max_used_search_results=5,
            response_language="eng"
        )
    )

    query_text = "Tell me about the fox"

    # Test Non-Streaming Query
    results = vectara.vectara_query(query_text, config)

    assert len(results) > 0
    assert all(isinstance(doc, tuple) and isinstance(doc[0], Document) for doc in
               results)

    # Test Streaming Query
    config.stream_response = True
    streamed_results = vectara.vectara_query(query_text, config)

    assert hasattr(streamed_results, "__iter__")

    streamed_chunks = list(streamed_results)  # Collect all streamed responses

    assert len(streamed_chunks) > 0
    assert any("answer" in chunk or "context" in chunk for chunk in
               streamed_chunks)

