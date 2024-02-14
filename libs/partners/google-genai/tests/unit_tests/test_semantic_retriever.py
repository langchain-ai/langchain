from unittest.mock import MagicMock, patch

import pytest

try:
    import google.ai.generativelanguage as genai

    has_google = True
except ImportError:
    has_google = False

from langchain.schema.document import Document

from langchain_google_genai import GoogleVectorStore

if has_google:
    from langchain_google_genai import _genai_extension as genaix

    # Make sure the tests do not hit actual production servers.
    genaix.set_config(
        genaix.Config(
            api_endpoint="No-such-endpoint-to-prevent-hitting-real-backend",
            testing=True,
        )
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
    mock_get_document.return_value = genai.Document(name="corpora/123/documents/456")

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
@patch("google.ai.generativelanguage.RetrieverServiceClient.batch_create_chunks")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_document")
def test_from_texts(
    mock_get_document: MagicMock,
    mock_batch_create_chunks: MagicMock,
) -> None:
    # Arrange
    # We will use a max requests per batch to be 2.
    # Then, we send 3 requests.
    # We expect to have 2 batches where the last batch has only 1 request.
    genaix._MAX_REQUEST_PER_CHUNK = 2
    mock_get_document.return_value = genai.Document(
        name="corpora/123/documents/456", display_name="My Document"
    )
    mock_batch_create_chunks.side_effect = [
        genai.BatchCreateChunksResponse(
            chunks=[
                genai.Chunk(name="corpora/123/documents/456/chunks/777"),
                genai.Chunk(name="corpora/123/documents/456/chunks/888"),
            ]
        ),
        genai.BatchCreateChunksResponse(
            chunks=[
                genai.Chunk(name="corpora/123/documents/456/chunks/999"),
            ]
        ),
    ]

    # Act
    store = GoogleVectorStore.from_texts(
        texts=[
            "Hello my baby",
            "Hello my honey",
            "Hello my ragtime gal",
        ],
        metadatas=[
            {"position": 100},
            {"position": 200},
            {"position": 300},
        ],
        corpus_id="123",
        document_id="456",
    )

    # Assert
    assert store.corpus_id == "123"
    assert store.document_id == "456"

    assert mock_batch_create_chunks.call_count == 2

    first_batch_request = mock_batch_create_chunks.call_args_list[0].args[0]
    assert first_batch_request == genai.BatchCreateChunksRequest(
        parent="corpora/123/documents/456",
        requests=[
            genai.CreateChunkRequest(
                parent="corpora/123/documents/456",
                chunk=genai.Chunk(
                    data=genai.ChunkData(string_value="Hello my baby"),
                    custom_metadata=[
                        genai.CustomMetadata(
                            key="position",
                            numeric_value=100,
                        ),
                    ],
                ),
            ),
            genai.CreateChunkRequest(
                parent="corpora/123/documents/456",
                chunk=genai.Chunk(
                    data=genai.ChunkData(string_value="Hello my honey"),
                    custom_metadata=[
                        genai.CustomMetadata(
                            key="position",
                            numeric_value=200,
                        ),
                    ],
                ),
            ),
        ],
    )

    second_batch_request = mock_batch_create_chunks.call_args_list[1].args[0]
    assert second_batch_request == genai.BatchCreateChunksRequest(
        parent="corpora/123/documents/456",
        requests=[
            genai.CreateChunkRequest(
                parent="corpora/123/documents/456",
                chunk=genai.Chunk(
                    data=genai.ChunkData(string_value="Hello my ragtime gal"),
                    custom_metadata=[
                        genai.CustomMetadata(
                            key="position",
                            numeric_value=300,
                        ),
                    ],
                ),
            ),
        ],
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


@pytest.mark.requires("google.ai.generativelanguage")
@patch("google.ai.generativelanguage.GenerativeServiceClient.generate_answer")
@patch("google.ai.generativelanguage.RetrieverServiceClient.query_corpus")
@patch("google.ai.generativelanguage.RetrieverServiceClient.get_corpus")
def test_aqa(
    mock_get_corpus: MagicMock,
    mock_query_corpus: MagicMock,
    mock_generate_answer: MagicMock,
) -> None:
    # Arrange
    mock_get_corpus.return_value = genai.Corpus(name="corpora/123")
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
    mock_generate_answer.return_value = genai.GenerateAnswerResponse(
        answer=genai.Candidate(
            content=genai.Content(parts=[genai.Part(text="42")]),
            grounding_attributions=[
                genai.GroundingAttribution(
                    content=genai.Content(
                        parts=[genai.Part(text="Meaning of life is 42.")]
                    ),
                    source_id=genai.AttributionSourceId(
                        grounding_passage=genai.AttributionSourceId.GroundingPassageId(
                            passage_id="corpora/123/documents/456/chunks/789",
                            part_index=0,
                        )
                    ),
                ),
            ],
            finish_reason=genai.Candidate.FinishReason.STOP,
        ),
        answerable_probability=0.7,
    )

    # Act
    store = GoogleVectorStore(corpus_id="123")
    aqa = store.as_aqa(answer_style=genai.GenerateAnswerRequest.AnswerStyle.EXTRACTIVE)
    response = aqa.invoke("What is the meaning of life?")

    # Assert
    assert response.answer == "42"
    assert response.attributed_passages == ["Meaning of life is 42."]
    assert response.answerable_probability == pytest.approx(0.7)

    request = mock_generate_answer.call_args.args[0]
    assert request.answer_style == genai.GenerateAnswerRequest.AnswerStyle.EXTRACTIVE
