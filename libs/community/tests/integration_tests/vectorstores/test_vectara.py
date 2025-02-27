import os
import re
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, cast

import pytest
import requests
from _pytest.logging import LogCaptureFixture
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from langchain_community.vectorstores import Vectara
from langchain_community.vectorstores.vectara import (
    ChainReranker,
    ChunkingStrategy,
    Citation,
    CorpusConfig,
    CustomerSpecificReranker,
    File,
    GenerationConfig,
    MmrReranker,
    NoneReranker,
    SearchConfig,
    TableExtractionConfig,
    UserFunctionReranker,
    VectaraQueryConfig,
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
def vectara() -> Generator[Vectara, None, None]:
    api_key = os.getenv("VECTARA_API_KEY")
    if not api_key:
        pytest.skip("VECTARA_API_KEY environment variable not set")
    vectara_instance = Vectara(vectara_api_key=api_key)

    yield vectara_instance

    cleanup_documents(vectara_instance, os.getenv("VECTARA_CORPUS_KEY"))


def cleanup_documents(vectara: Vectara, corpus_key: str | None) -> None:
    """
    Fetch all documents from the corpus and delete them after tests are completed.
    """

    url = f"https://api.vectara.io/v2/corpora/{corpus_key}/documents"
    headers = {"Accept": "application/json", "x-api-key": str(vectara._vectara_api_key)}

    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        return

    data = response.json()
    document_ids = [doc["id"] for doc in data.get("documents", [])]

    if not document_ids:
        return

    vectara.delete(ids=document_ids, corpus_key=corpus_key)


@pytest.fixture(scope="module")
def corpus_key() -> str:
    corpus_key = os.getenv("VECTARA_CORPUS_KEY")
    if not corpus_key:
        pytest.skip("VECTARA_CORPUS_KEY environment variable not set")
    return corpus_key


@pytest.fixture(scope="module")
def add_test_doc(vectara: Vectara, corpus_key: str) -> None:
    texts = [
        "The quick brown fox jumps over the lazy dog."
        "The lazy dog sleeps while the quick brown fox jumps."
        "A fox is quick and brown in color."
    ]
    vectara.add_texts(
        texts=texts,
        corpus_key=corpus_key,
        doc_metadata={"url": "https://www.example.com"},
    )


@pytest.fixture
def temp_files(tmp_path: Path) -> Dict[str, str]:
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


def test_initialization(vectara: Vectara) -> None:
    """Check that the instance is initialized properly."""
    api_key = os.getenv("VECTARA_API_KEY")
    assert vectara._vectara_api_key == api_key


def test_get_post_headers(vectara: Vectara) -> None:
    """Ensure headers are correctly formed."""
    headers = vectara._get_post_headers()
    assert "x-api-key" in headers
    assert "Content-Type" in headers


def test_add_texts(vectara: Vectara, corpus_key: str) -> None:
    """Index texts and verify they are indexed successfully as separate documents."""

    # Test adding texts with core document type
    texts_core = [
        "This is a test document of type core.",
        "This is another test document.",
        "This is a third test document.",
    ]
    metadatas_core = [{"source": "test1"}, {"source": "test2"}, {"source": "test3"}]
    doc_ids_core = vectara.add_texts(
        texts=texts_core,
        metadatas=metadatas_core,
        doc_type="core",
        corpus_key=corpus_key,
    )
    # Expect one ID per text
    assert len(doc_ids_core) == len(texts_core)
    for doc_id in doc_ids_core:
        assert isinstance(doc_id, str)

    # Test adding texts with structured document type
    texts_structured = [
        "This is a test document of type structured.",
        "This is another test document.",
        "This is a third test document.",
    ]
    metadatas_structured = [
        {"source": "test1"},
        {"source": "test2"},
        {"source": "test3"},
    ]
    doc_ids_structured = vectara.add_texts(
        texts=texts_structured,
        metadatas=metadatas_structured,
        doc_type="structured",
        corpus_key=corpus_key,
    )
    # Expect one ID per text
    assert len(doc_ids_structured) == len(texts_structured)
    for doc_id in doc_ids_structured:
        assert isinstance(doc_id, str)


def test_text_without_corpus_key(vectara: Vectara) -> None:
    """Ensure that Vectara add_text was called."""

    texts = ["Testing as generic VectorStore."]
    metadatas = [{"source": "generic_test"}]

    with pytest.raises(ValueError, match="Missing required parameter: 'corpus_key'."):
        vectara.add_texts(texts=texts, metadatas=metadatas, doc_type="structured")


def test_add_documents_via_vectara_retriever(vectara: Vectara, corpus_key: str) -> None:
    """
    Test that VectaraRetriever.add_documents indexes each Document as a
    separate document. This test creates several Document objects, then calls
    add_documents on the retriever, verifying that the returned document IDs match the
    number of input documents and are strings.
    """
    # Create a list of Document objects
    docs = [
        Document(page_content="Document 1 content", metadata={"source": "test"}),
        Document(page_content="Document 2 content", metadata={"source": "test"}),
        Document(page_content="Document 3 content", metadata={"source": "test"}),
    ]

    retriever = vectara.as_retriever(
        config=VectaraQueryConfig(
            search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)])
        )
    )

    doc_ids = retriever.add_documents(
        docs, corpus_key=corpus_key, doc_type="structured"
    )

    assert len(doc_ids) == len(docs)
    for doc_id in doc_ids:
        assert isinstance(doc_id, str)


def test_get_relevant_documents(vectara: Vectara, corpus_key: str) -> None:
    """Test _get_relevant_documents in VectaraRetriever returns a
    list of Document instances."""

    retriever = vectara.as_retriever(
        config=VectaraQueryConfig(
            search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)])
        )
    )

    # Dummy callback manager for testing purposes
    class DummyCallbackManager(CallbackManagerForRetrieverRun):
        """A subclass of CallbackManagerForRetrieverRun with default args."""

        def __init__(
            self,
            run_id: uuid.UUID = uuid.uuid4(),
            handlers: Optional[List[Any]] = None,
            inheritable_handlers: Optional[List[Any]] = None,
        ) -> None:
            super().__init__(
                run_id=run_id,
                handlers=handlers or [],
                inheritable_handlers=inheritable_handlers or [],
            )

    dummy_run_manager = DummyCallbackManager()
    docs = retriever._get_relevant_documents(
        "What is a fox?", run_manager=dummy_run_manager
    )

    assert isinstance(docs, list)
    for doc in docs:
        assert isinstance(doc, Document)


def test_from_texts(corpus_key: str) -> None:
    """Test that Vectara.from_texts returns a Vectara instance and indexes documents."""
    texts = [
        "This is a test document for from_texts.",
        "Another test document for from_texts.",
    ]
    metadatas = [{"source": "test1"}, {"source": "test2"}]
    api_key = os.getenv("VECTARA_API_KEY")
    if not api_key:
        pytest.skip("VECTARA_API_KEY environment variable not set")

    vectara_instance = Vectara.from_texts(
        texts=texts,
        metadatas=metadatas,
        vectara_api_key=api_key,
        corpus_key=corpus_key,
        doc_type="core",
    )

    assert isinstance(vectara_instance, Vectara)

    config_search = SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)])

    results = vectara_instance.similarity_search("test document", search=config_search)

    assert isinstance(results, list)


def test_from_documents(corpus_key: str) -> None:
    """
    Test that Vectara.from_documents returns a Vectara instance
    and indexes documents using from_texts (which is called under the hood).
    """
    docs = [
        Document(page_content="Document 1 content", metadata={"source": "test"}),
        Document(page_content="Document 2 content", metadata={"source": "test"}),
        Document(page_content="Document 3 content", metadata={"source": "test"}),
    ]

    api_key = os.getenv("VECTARA_API_KEY")
    if not api_key:
        pytest.skip("VECTARA_API_KEY environment variable not set")

    vectara_instance = Vectara.from_documents(
        documents=docs,
        embedding=cast(Embeddings, None),
        vectara_api_key=api_key,
        corpus_key=corpus_key,
        doc_type="structured",
        doc_metadata={"extra": "value"},
    )

    assert isinstance(vectara_instance, Vectara)

    config_search = SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)])
    results = vectara_instance.similarity_search("Document", search=config_search)

    assert isinstance(results, list)


def test_query_with_citation(
    vectara: Vectara, corpus_key: str, add_test_doc: None
) -> None:
    """
    Test that when a citation configuration is provided in the query generation config,
    the summary output (last document) includes citation formatting as expected.
    """
    citation_config = Citation(
        style="markdown", url_pattern="{doc.url}", text_pattern="(source)"
    )

    generation_config = GenerationConfig(
        max_used_search_results=7,
        response_language="eng",
        generation_preset_name="vectara-summary-ext-24-05-med-omni",
        enable_factual_consistency_score=True,
        citations=citation_config,
    )

    config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
        generation=generation_config,
        stream_response=False,
        save_history=False,
        chat=False,
    )

    query_text = "What color is the fox?"
    results = vectara.vectara_query(query_text, config)

    assert isinstance(results, list)
    assert len(results) > 0

    summary_doc, score = results[-1]
    assert isinstance(summary_doc, Document)

    summary_content = summary_doc.page_content
    assert "(source)" in summary_content
    assert "https://www.example.com" in summary_content

    # test numeric citations
    citation_config = Citation(style="numeric")
    if config.generation:
        config.generation.citations = citation_config

    query_text = "What color is the fox?"
    results = vectara.vectara_query(query_text, config)

    assert isinstance(results, list)
    assert len(results) > 0

    summary_doc, score = results[-1]
    assert isinstance(summary_doc, Document)

    summary_content = summary_doc.page_content
    assert re.search(r"\[\d+\]", summary_content)


def test_add_files_text(
    vectara: Vectara, corpus_key: str, temp_files: Dict[str, str]
) -> None:
    """Test uploading a TXT file for indexing."""
    file_obj = File(file_path=temp_files["test.txt"], metadata={"source": "text_file"})
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)


def test_add_files_pdf(vectara: Vectara, corpus_key: str, tmp_path: Path) -> None:
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
    with open(test_pdf_path, "wb") as f:
        f.write(pdf_content)

    file_obj = File(
        file_path=str(test_pdf_path),
        metadata={"source": "pdf_file"},
        table_extraction_config=TableExtractionConfig(extract_tables=True),
    )
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)


def test_add_files_pptx(
    vectara: Vectara, corpus_key: str, temp_files: Dict[str, str]
) -> None:
    """Test uploading a PPTX file for indexing."""
    file_obj = File(file_path=temp_files["test.pptx"], metadata={"source": "pptx_file"})
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)


def test_add_files_docx(
    vectara: Vectara, corpus_key: str, temp_files: Dict[str, str]
) -> None:
    """Test uploading a DOCX file for indexing."""
    file_obj = File(
        file_path=temp_files["test.docx"],
        metadata={"source": "docx_file"},
        chunking_strategy=ChunkingStrategy(max_chars_per_chunk=500),
    )
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)


def test_add_files_markdown(
    vectara: Vectara, corpus_key: str, temp_files: Dict[str, str]
) -> None:
    """Test uploading a Markdown (MD) file for indexing."""
    file_obj = File(
        file_path=temp_files["test.md"], metadata={"source": "markdown_file"}
    )
    doc_ids = vectara.add_files([file_obj], corpus_key)

    assert len(doc_ids) > 0
    assert isinstance(doc_ids[0], str)


def test_add_files_multiple(vectara: Vectara, corpus_key: str, tmp_path: Path) -> None:
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


def test_add_files_missing(
    vectara: Vectara, corpus_key: str, caplog: LogCaptureFixture
) -> None:
    """Test attempting to upload a non-existent file (should log an error)."""
    non_existent_file = File(
        file_path="invalid_path.txt", metadata={"source": "missing_file"}
    )

    with caplog.at_level("ERROR"):
        doc_ids = vectara.add_files([non_existent_file], corpus_key)

    assert len(doc_ids) == 0
    assert "File invalid_path.txt does not exist" in caplog.text


def test_add_files_empty_list(vectara: Vectara, corpus_key: str) -> None:
    """Test calling add_files with an empty list (should return empty list)."""
    doc_ids = vectara.add_files([], corpus_key)
    assert doc_ids == []


def test_similarity_search(
    vectara: Vectara, corpus_key: str, add_test_doc: None
) -> None:
    """Run a similarity search query."""
    # Test basic similarity search
    results = vectara.similarity_search(
        query="What color is the fox?",
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
    )
    assert len(results) > 0
    assert isinstance(results[0], Document)

    # Test similarity search with scores
    results_with_scores = vectara.similarity_search_with_score(
        query="What color is the fox?",
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
    )
    assert len(results_with_scores) > 0
    assert isinstance(results_with_scores[0], tuple)
    assert isinstance(results_with_scores[0][0], Document)
    assert isinstance(results_with_scores[0][1], float)


def test_mmr_search(vectara: Vectara, corpus_key: str, add_test_doc: None) -> None:
    query = "What color is the fox?"
    results = vectara.max_marginal_relevance_search(
        query=query,
        fetch_k=5,
        lambda_mult=0.5,
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
    )
    assert len(results) > 0
    assert isinstance(results[0], Document)


def test_customer_specific_reranker(
    vectara: Vectara, corpus_key: str, add_test_doc: None
) -> None:
    """Test a search query with CustomerSpecificReranker."""
    reranker = CustomerSpecificReranker(reranker_id="rnk_272725719", limit=3)
    config_search = SearchConfig(
        corpora=[CorpusConfig(corpus_key=corpus_key)], reranker=reranker
    )

    results = vectara.similarity_search("What color is the fox?", search=config_search)
    assert len(results) > 0
    assert isinstance(results[0], Document)


def test_none_reranker(vectara: Vectara, corpus_key: str, add_test_doc: None) -> None:
    """Test a search query without reranking."""
    reranker = NoneReranker()
    config_search = SearchConfig(
        corpora=[CorpusConfig(corpus_key=corpus_key)], reranker=reranker
    )

    results = vectara.similarity_search("What color is the fox?", search=config_search)
    assert len(results) > 0
    assert isinstance(results[0], Document)


def test_user_function_reranker(
    vectara: Vectara, corpus_key: str, add_test_doc: None
) -> None:
    """Test a search query with a user-defined function reranker."""
    reranker = UserFunctionReranker(
        user_function="if (get('$.score') < 0.1) null else get('$.score') + 1"
    )
    config_search = SearchConfig(
        corpora=[CorpusConfig(corpus_key=corpus_key)], reranker=reranker
    )

    results = vectara.similarity_search("What color is the fox?", search=config_search)
    assert len(results) > 0
    assert isinstance(results[0], Document)


def test_chain_reranker(vectara: Vectara, corpus_key: str, add_test_doc: None) -> None:
    """Test a search query with multiple rerankers applied sequentially."""
    reranker = ChainReranker(
        rerankers=[
            CustomerSpecificReranker(reranker_id="rnk_272725719", limit=10),
            MmrReranker(diversity_bias=0.3, limit=5),
        ]
    )

    config_search = SearchConfig(
        corpora=[CorpusConfig(corpus_key=corpus_key)], reranker=reranker
    )

    results = vectara.similarity_search("What color is the fox?", search=config_search)
    assert len(results) > 0
    assert isinstance(results[0], Document)


def test_delete_documents(vectara: Vectara, corpus_key: str) -> None:
    texts = ["This is a test document to be deleted."]
    doc_ids = vectara.add_texts(texts=texts, corpus_key=corpus_key)

    success = vectara.delete(ids=doc_ids, corpus_key=corpus_key)
    assert success is True


def test_vectara_as_rag(vectara: Vectara, corpus_key: str, add_test_doc: None) -> None:
    config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
        generation=GenerationConfig(max_used_search_results=5, response_language="eng"),
    )

    rag = vectara.as_rag(config)

    result = rag.invoke("What color is the fox?")
    assert "question" in result
    assert "answer" in result
    assert "context" in result


def test_streaming(vectara: Vectara, corpus_key: str, add_test_doc: None) -> None:
    config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
        generation=GenerationConfig(max_used_search_results=5),
        stream_response=True,
    )

    rag = vectara.as_rag(config)
    chunks = list(rag.stream("What color is the fox?"))

    assert len(chunks) > 0
    assert any("question" in chunk for chunk in chunks)
    assert any("answer" in chunk for chunk in chunks)
    assert any("context" in chunk for chunk in chunks)


def test_as_chat(vectara: Vectara, corpus_key: str, add_test_doc: None) -> None:
    config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]), chat=True
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


def test_vectara_query(vectara: Vectara, corpus_key: str, add_test_doc: None) -> None:
    """Test vectara_query"""

    config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
        generation=GenerationConfig(max_used_search_results=5, response_language="eng"),
    )

    query_text = "Tell me about the fox"

    # Test Non-Streaming Query
    results = list(vectara.vectara_query(query_text, config))

    assert len(results) > 0
    assert all(
        isinstance(doc, tuple) and isinstance(doc[0], Document) for doc in results
    )


def test_vectara_query_chat(
    vectara: Vectara, corpus_key: str, add_test_doc: None
) -> None:
    """Test Vectara query in chat mode and ensure conversation continuity."""

    config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
        chat=True,
    )

    response_list = list(vectara.vectara_query("Tell me about the fox", config))

    assert len(response_list) > 0

    # Last document should contain the answer and chat_convo_id
    last_doc, score = response_list[-1]
    assert isinstance(last_doc, Document)
    assert "chat_convo_id" in last_doc.metadata
    assert isinstance(last_doc.metadata["chat_convo_id"], str)

    chat_id = last_doc.metadata["chat_convo_id"]

    # Continue the conversation with a follow-up question
    followup_config = VectaraQueryConfig(
        search=SearchConfig(corpora=[CorpusConfig(corpus_key=corpus_key)]),
        chat=True,
        chat_conv_id=chat_id,
    )

    # Convert the followup response to a list
    followup_response_list = list(
        vectara.vectara_query("What color is the fox?", followup_config)
    )

    assert len(followup_response_list) > 0

    # Last document should contain the follow-up answer and same chat ID
    last_doc_followup, score_followup = followup_response_list[-1]
    assert isinstance(last_doc_followup, Document)
    assert "chat_convo_id" in last_doc_followup.metadata
    assert last_doc_followup.metadata["chat_convo_id"] == chat_id


def test_get_by_ids(vectara: Vectara, corpus_key: str) -> None:
    """
    Integration test for 'get_document' and 'get_by_ids'.

    1. Adds documents via 'add_texts'.
    2. Calls 'get_by_ids' to fetch them from Vectara.
    3. Verifies the retrieved documents match the original text/metadata.
    """

    texts = ["This is the content of Document A.", "Here is the text for Document B."]
    metadatas = [
        {"test": "testA", "custom": "A_metadata"},
        {"test": "testB", "custom": "B_metadata"},
    ]

    doc_ids = vectara.add_texts(
        texts=texts, metadatas=metadatas, corpus_key=corpus_key, doc_type="structured"
    )
    assert len(doc_ids) == len(texts)

    retrieved_docs = vectara.get_by_ids(doc_ids, corpus_key=corpus_key)
    assert len(retrieved_docs) == len(doc_ids)

    # Verify the merged text and metadata in each retrieved Document
    # We expect 1:1 order with doc_ids -> retrieved_docs, but if not guaranteed,
    # we can match by content
    for idx, doc in enumerate(retrieved_docs):
        assert isinstance(doc, Document)
        # Confirm some or all of the text is present
        assert texts[idx] in doc.page_content, (
            f"Expected content '{texts[idx]}' to be in the retrieved doc page_content."
        )
        # Confirm top-level metadata was merged
        # Our doc metadata is stored in doc.metadata.
        # If 'parts' metadata was also merged, check here.
        # e.g., 'source' or 'custom' keys from the original input:
        assert doc.metadata.pop("source") == "langchain"
        assert doc.metadata.get("test") == metadatas[idx]["test"]
        assert doc.metadata.get("custom") == metadatas[idx]["custom"]
