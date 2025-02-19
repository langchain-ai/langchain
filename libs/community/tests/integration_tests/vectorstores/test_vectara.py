import os
import uuid

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
    ChunkingStrategy,
    CustomerSpecificReranker,
    UserFunctionReranker,
    NoneReranker,
    ChainReranker
)

from langchain_core.vectorstores import VectorStore

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

@pytest.fixture
def temp_files(tmp_path):
    """Fixture to create multiple test files in various formats."""
    file_contents = {
        "test.txt": "This is a test file content.",
        "test.pptx": "PPTX FILE CONTENT",
        "test.docx": "DOCX FILE CONTENT",
        "test.md": "# This is a Markdown file\n\nSome sample text.",
    }

    created_files = {}
    for filename, content in file_contents.items():
        file_path = tmp_path / filename
        file_path.write_text(content)
        created_files[filename] = str(file_path)

    return created_files

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

def test_call_add_text_using_vectorstore():
    """Ensure that add_text can be used generic VectorStore."""
    api_key = os.getenv("VECTARA_API_KEY")
    corpus_key = os.getenv("VECTARA_CORPUS_KEY")

    if not api_key or not corpus_key:
        pytest.skip("VECTARA_API_KEY or VECTARA_CORPUS_KEY environment variable not set")

    vectara: VectorStore = Vectara(vectara_api_key=api_key)  # Store in VectorStore type

    texts = ["Testing as generic VectorStore."]
    metadatas = [{"source": "generic_test"}]

    doc_ids = vectara.add_texts(
        texts=texts,
        metadatas=metadatas,
        corpus_key=corpus_key,
        doc_type="structured"
    )

    assert len(doc_ids) == 1
    assert isinstance(doc_ids[0], str)


def test_add_files_text(vectara, corpus_key, temp_files):
    """Test uploading a TXT file for indexing."""
    file_obj = File(file_path=temp_files["test.txt"], metadata={"source": "text_file"})
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)

def test_add_files_pdf(vectara, corpus_key, tmp_path):
    """Test uploading a PDF file for indexing."""
    test_pdf_path = tmp_path / "test.pdf"

    pdf_content = (
        b"%PDF-1.7\n"  # PDF version header
        b"1 0 obj\n"
        b"<< /Type /Catalog /Pages 2 0 R >>\n"
        b"endobj\n"

        b"2 0 obj\n"
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>\n"
        b"endobj\n"

        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>\n"
        b"endobj\n"

        b"4 0 obj\n"
        b"<< /Type /Font /Subtype /Type1 /Name /F1 /BaseFont /Helvetica >>\n"
        b"endobj\n"

        b"5 0 obj\n"
        b"<< /Length 44 >>\n"
        b"stream\n"
        b"BT\n"
        b"/F1 24 Tf\n"
        b"100 700 Td\n"
        b"(Hello, World!) Tj\n"
        b"ET\n"
        b"endstream\n"
        b"endobj\n"

        b"xref\n"
        b"0 6\n"
        b"0000000000 65535 f\n"
        b"0000000010 00000 n\n"
        b"0000000060 00000 n\n"
        b"0000000120 00000 n\n"
        b"0000000250 00000 n\n"
        b"0000000340 00000 n\n"

        b"trailer\n"
        b"<< /Size 6 /Root 1 0 R >>\n"
        b"startxref\n"
        b"440\n"
        b"%%EOF"
    )

    # Write PDF to file
    with open(test_pdf_path, 'wb') as f:
        f.write(pdf_content)

    file_obj = File(
        file_path=str(test_pdf_path),
        metadata={"source": "pdf_file"},
        table_extraction_config=TableExtractionConfig(extract_tables=True),
    )
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)


def test_add_files_pptx(vectara, corpus_key, temp_files):
    """Test uploading a PPTX file for indexing."""
    file_obj = File(file_path=temp_files["test.pptx"], metadata={"source": "pptx_file"})
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)

def test_add_files_docx(vectara, corpus_key, temp_files):
    """Test uploading a DOCX file for indexing."""
    file_obj = File(
        file_path=temp_files["test.docx"],
        metadata={"source": "docx_file"},
        chunking_strategy=ChunkingStrategy(max_chars_per_chunk=500),
    )
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)

def test_add_files_markdown(vectara, corpus_key, temp_files):
    """Test uploading a Markdown (MD) file for indexing."""
    file_obj = File(file_path=temp_files["test.md"], metadata={"source": "markdown_file"})
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)

def test_add_files_multiple(vectara, corpus_key, tmp_path):
    """Test uploading multiple file types at once."""
    unique_id = uuid.uuid4().hex

    text_file = tmp_path / f"test_{unique_id}.txt"
    docx_file = tmp_path / f"test_{unique_id}.docx"

    text_file.write_text("This is a test text file.")
    docx_file.write_text("This is a test DOCX file.")

    files_list = [
        File(file_path=str(text_file), metadata={"source": "multi_text"}),
        File(file_path=str(docx_file), metadata={"source": "multi_docx"}),
    ]

    doc_ids = vectara.add_files(files_list, corpus_key)

    assert len(doc_ids) == len(files_list)
    assert all(isinstance(doc_id, str) for doc_id in doc_ids)

def test_add_files_missing(vectara, corpus_key, caplog):
    """Test attempting to upload a non-existent file (should log an error)."""
    non_existent_file = File(file_path="invalid_path.txt", metadata={"source": "missing_file"})

    with caplog.at_level("ERROR"):
        doc_ids = vectara.add_files([non_existent_file], corpus_key)

    assert len(doc_ids) == 0
    assert "File invalid_path.txt does not exist" in caplog.text

def test_add_files_empty_list(vectara, corpus_key):
    """Test calling add_files with an empty list (should return empty list)."""
    doc_ids = vectara.add_files([], corpus_key)
    assert doc_ids == []

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

def test_customer_specific_reranker(vectara, corpus_key, add_test_doc):
    """Test a search query with CustomerSpecificReranker."""
    reranker = CustomerSpecificReranker(reranker_id="rnk_272725719", limit=3)
    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)],
            reranker=reranker
        )
    )

    results = vectara.similarity_search("What color is the fox?", search=config.search)
    assert len(results) > 0
    assert isinstance(results[0], Document)

def test_none_reranker(vectara, corpus_key, add_test_doc):
    """Test a search query without reranking."""
    reranker = NoneReranker()
    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)],
            reranker=reranker
        )
    )

    results = vectara.similarity_search("What color is the fox?", search=config.search)
    assert len(results) > 0
    assert isinstance(results[0], Document)

def test_user_function_reranker(vectara, corpus_key, add_test_doc):
    """Test a search query with a user-defined function reranker."""
    reranker = UserFunctionReranker(user_function="if (get('$.score') < 0.1) null else get('$.score') + 1")
    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)],
            reranker=reranker
        )
    )

    results = vectara.similarity_search("What color is the fox?", search=config.search)
    assert len(results) > 0
    assert isinstance(results[0], Document)


def test_chain_reranker(vectara, corpus_key, add_test_doc):
    """Test a search query with multiple rerankers applied sequentially."""
    reranker = ChainReranker(
        rerankers=[
            MmrReranker(diversity_bias=0.3, limit=10),
            CustomerSpecificReranker( reranker_id="rnk_272725719", limit=5)
        ]
    )

    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)],
            reranker=reranker
        )
    )

    results = vectara.similarity_search("What color is the fox?", search=config.search)
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
            max_used_search_results=5,
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
            max_used_search_results=5
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
    chat_id = result["chat_id"]

    assert "question" in result
    assert "answer" in result
    assert "chat_id" in result
    assert config.chat_conv_id == chat_id

    chat = vectara.as_chat(config)
    result = chat.invoke("What color is the fox?")

    assert "question" in result
    assert "answer" in result
    assert "chat_id" in result
    assert config.chat_conv_id == result["chat_id"]


def test_vectara_query(vectara, corpus_key, add_test_doc):
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

def test_vectara_query_chat(vectara, corpus_key, add_test_doc):
    """Test Vectara query in chat mode and ensure conversation continuity."""

    config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        ),
        chat=True
    )

    response = vectara.vectara_query("Tell me about the fox", config)

    assert len(response) > 0

    # Last document should contain the answer and chat_convo_id
    last_doc, score = response[-1]

    assert isinstance(last_doc, Document)
    assert "chat_convo_id" in last_doc.metadata
    assert isinstance(last_doc.metadata["chat_convo_id"], str)

    chat_id = last_doc.metadata["chat_convo_id"]

    # Step 3: Continue the conversation with a follow-up question
    followup_config = VectaraQueryConfig(
        search=SearchConfig(
            corpora=[CorpusConfig(corpus_key=corpus_key)]
        ),
        chat=True,
        chat_conv_id=chat_id
    )

    followup_response = vectara.vectara_query("What color is the fox?", followup_config)

    # Ensure there are results
    assert len(followup_response) > 0

    # Last document should contain the follow-up answer and same chat ID
    last_doc_followup, score_followup = followup_response[-1]

    assert isinstance(last_doc_followup, Document)
    assert "chat_convo_id" in last_doc_followup.metadata
    assert last_doc_followup.metadata["chat_convo_id"] == chat_id
