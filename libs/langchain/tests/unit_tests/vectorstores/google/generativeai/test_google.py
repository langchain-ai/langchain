from unittest.mock import MagicMock, patch

import pytest

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from langchain.schema.document import Document
from langchain.vectorstores.google.generativeai import GoogleVectorStore

if has_google:
    import langchain.vectorstores.google.generativeai.genai_extension as genaix

    # Make sure the tests do not hit actual production servers.
    genaix.set_defaults(
        genaix.Config(
            api_endpoint="No-such-endpoint-to-prevent-hitting-real-backend")
    )


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_load_corpus(mock_get_corpus: MagicMock) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")

    # Act
    store = GoogleVectorStore(corpus_id="123")

    # Assert
    assert store.name == "corpora/123"
    assert store.corpus_id == "123"
    assert store.document_id is None


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_document")
def test_load_document(mock_get_document: MagicMock) -> None:
    # Arrange
    mock_get_document.return_value = genai.Document(
        name="corpora/123/documents/456")

    # Act
    store = GoogleVectorStore(corpus_id="123", document_id="456")

    # Assert
    assert store.name == "corpora/123/documents/456"
    assert store.corpus_id == "123"
    assert store.document_id == "456"


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_corpus")
def test_create_corpus(
    mock_create_corpus: MagicMock, mock_get_corpus: MagicMock
) -> None:
    # Arrange
    fake_corpus = genai.Corpus(name="corpora/123", display_name="My Corpus")
    mock_create_corpus.return_value = fake_corpus
    mock_get_corpus.return_value = fake_corpus

    # Act
    store = GoogleVectorStore.create_corpus(display_name="My Corpus")

    # Assert
    assert store.name == "corpora/123"
    assert store.corpus_id == "123"
    assert store.document_id is None

    assert mock_create_corpus.call_count == 1

    create_request = mock_create_corpus.call_args.args[0]
    assert create_request.corpus.name == ""
    assert create_request.corpus.display_name == "My Corpus"

    get_request = mock_get_corpus.call_args.args[0]
    assert get_request.name == "corpora/123"


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_document")
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_document")
def test_create_document(
    mock_create_document: MagicMock, mock_get_document: MagicMock
) -> None:
    # Arrange
    fake_document = genai.Document(
        name="corpora/123/documents/456", display_name="My Document"
    )
    mock_create_document.return_value = fake_document
    mock_get_document.return_value = fake_document

    # Act
    store = GoogleVectorStore.create_document(
        corpus_id="123", display_name="My Document"
    )

    # Assert
    assert store.name == "corpora/123/documents/456"
    assert store.corpus_id == "123"
    assert store.document_id == "456"

    assert mock_create_document.call_count == 1

    create_request = mock_create_document.call_args.args[0]
    assert create_request.parent == "corpora/123"
    assert create_request.document.name == ""
    assert create_request.document.display_name == "My Document"

    get_request = mock_get_document.call_args.args[0]
    assert get_request.name == "corpora/123/documents/456"


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.create_chunk")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_document")
def test_from_texts(
    mock_get_document: MagicMock,
    mock_create_chunk: MagicMock,
) -> None:
    # Arrange
    mock_get_document.return_value = genai.Document(
        name="corpora/123/documents/456", display_name="My Document"
    )

    # Act
    store = GoogleVectorStore.from_texts(
        texts=[
            "Hello, my darling",
            "Goodbye, my baby",
        ],
        metadatas=[
            {"author": "Alice"},
            {"author": "Bob"},
        ],
        corpus_id="123",
        document_id="456",
    )

    # Assert
    assert store.corpus_id == "123"
    assert store.document_id == "456"

    assert mock_create_chunk.call_count == 2
    create_chunk_requests = mock_create_chunk.call_args_list

    first_create_chunk_request = create_chunk_requests[0].args[0]
    assert first_create_chunk_request == genai.CreateChunkRequest(
        parent="corpora/123/documents/456",
        chunk=genai.Chunk(
            data=genai.ChunkData(string_value="Hello, my darling"),
            custom_metadata=[
                genai.CustomMetadata(
                    key="author",
                    string_value="Alice",
                ),
            ],
        ),
    )

    second_create_chunk_request = create_chunk_requests[1].args[0]
    assert second_create_chunk_request == genai.CreateChunkRequest(
        parent="corpora/123/documents/456",
        chunk=genai.Chunk(
            data=genai.ChunkData(string_value="Goodbye, my baby"),
            custom_metadata=[
                genai.CustomMetadata(
                    key="author",
                    string_value="Bob",
                ),
            ],
        ),
    )


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.query_corpus")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_similarity_search_with_score_on_corpus(
    mock_get_corpus: MagicMock,
    mock_query_corpus: MagicMock,
) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(
        name="corpora/123", display_name="My Corpus"
    )
    mock_query_corpus.return_value = genai.QueryCorpusResponse(
        relevant_chunks=[
            genai.RelevantChunk(
                chunk=genai.Chunk(
                    name="corpora/123/documents/456/chunks/789",
                    data=genai.ChunkData(string_value="42"),
                ),
                chunk_relevance_score=0.9,
            )
        ]
    )

    # Act
    store = GoogleVectorStore(corpus_id="123")
    documents_with_scores = store.similarity_search_with_score(
        query="What is the meaning of life?",
        k=3,
        filter={
            "author": "Arthur Schopenhauer",
            "year": 1818,
        },
    )

    # Assert
    assert len(documents_with_scores) == 1
    document, relevant_score = documents_with_scores[0]
    assert document == Document(page_content="42")
    assert relevant_score == pytest.approx(0.9)

    assert mock_query_corpus.call_count == 1
    query_corpus_request = mock_query_corpus.call_args.args[0]
    assert query_corpus_request == genai.QueryCorpusRequest(
        name="corpora/123",
        query="What is the meaning of life?",
        metadata_filters=[
            genai.MetadataFilter(
                key="author",
                conditions=[
                    genai.Condition(
                        operation=genai.Condition.Operator.EQUAL,
                        string_value="Arthur Schopenhauer",
                    )
                ],
            ),
            genai.MetadataFilter(
                key="year",
                conditions=[
                    genai.Condition(
                        operation=genai.Condition.Operator.EQUAL,
                        numeric_value=1818,
                    )
                ],
            ),
        ],
        results_count=3,
    )


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.query_document")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_document")
def test_similarity_search_with_score_on_document(
    mock_get_document: MagicMock,
    mock_query_document: MagicMock,
) -> None:
    # Arrange
    mock_get_document.return_value = genai.Document(
        name="corpora/123/documents/456", display_name="My Document"
    )
    mock_query_document.return_value = genai.QueryCorpusResponse(
        relevant_chunks=[
            genai.RelevantChunk(
                chunk=genai.Chunk(
                    name="corpora/123/documents/456/chunks/789",
                    data=genai.ChunkData(string_value="42"),
                ),
                chunk_relevance_score=0.9,
            )
        ]
    )

    # Act
    store = GoogleVectorStore(corpus_id="123", document_id="456")
    documents_with_scores = store.similarity_search_with_score(
        query="What is the meaning of life?",
        k=3,
        filter={
            "author": "Arthur Schopenhauer",
            "year": 1818,
        },
    )

    # Assert
    assert len(documents_with_scores) == 1
    document, relevant_score = documents_with_scores[0]
    assert document == Document(page_content="42")
    assert relevant_score == pytest.approx(0.9)

    assert mock_query_document.call_count == 1
    query_document_request = mock_query_document.call_args.args[0]
    assert query_document_request == genai.QueryDocumentRequest(
        name="corpora/123/documents/456",
        query="What is the meaning of life?",
        metadata_filters=[
            genai.MetadataFilter(
                key="author",
                conditions=[
                    genai.Condition(
                        operation=genai.Condition.Operator.EQUAL,
                        string_value="Arthur Schopenhauer",
                    )
                ],
            ),
            genai.MetadataFilter(
                key="year",
                conditions=[
                    genai.Condition(
                        operation=genai.Condition.Operator.EQUAL,
                        numeric_value=1818,
                    )
                ],
            ),
        ],
        results_count=3,
    )


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.RetrieverServiceClient.delete_chunk")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_delete(
    mock_get_corpus: MagicMock,
    mock_delete_chunk: MagicMock,
) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")

    # Act
    store = GoogleVectorStore(corpus_id="123")
    store.delete(
        ids=[
            "corpora/123/documents/456/chunks/1001",
            "corpora/123/documents/456/chunks/1002",
        ]
    )

    # Assert
    assert mock_delete_chunk.call_count == 2
    delete_chunk_requests = mock_delete_chunk.call_args_list

    delete_chunk_request_1 = delete_chunk_requests[0].args[0]
    assert delete_chunk_request_1 == genai.DeleteChunkRequest(
        name="corpora/123/documents/456/chunks/1001",
    )

    delete_chunk_request_2 = delete_chunk_requests[1].args[0]
    assert delete_chunk_request_2 == genai.DeleteChunkRequest(
        name="corpora/123/documents/456/chunks/1002",
    )
