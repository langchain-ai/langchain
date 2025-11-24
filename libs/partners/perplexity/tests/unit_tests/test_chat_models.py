from typing import Any, cast
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, BaseMessage
from pytest_mock import MockerFixture

from langchain_perplexity import ChatPerplexity
from langchain_perplexity.chat_models import _create_usage_metadata


def test_perplexity_model_name_param() -> None:
    llm = ChatPerplexity(model="foo")
    assert llm.model == "foo"


def test_perplexity_model_kwargs() -> None:
    llm = ChatPerplexity(model="test", model_kwargs={"foo": "bar"})
    assert llm.model_kwargs == {"foo": "bar"}


def test_perplexity_initialization() -> None:
    """Test perplexity initialization."""
    # Verify that chat perplexity can be initialized using a secret key provided
    # as a parameter rather than an environment variable.
    for model in [
        ChatPerplexity(
            model="test", timeout=1, api_key="test", temperature=0.7, verbose=True
        ),
        ChatPerplexity(
            model="test",
            request_timeout=1,
            pplx_api_key="test",
            temperature=0.7,
            verbose=True,
        ),
    ]:
        assert model.request_timeout == 1
        assert (
            model.pplx_api_key is not None
            and model.pplx_api_key.get_secret_value() == "test"
        )


def test_perplexity_stream_includes_citations(mocker: MockerFixture) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)
    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {"content": "Perplexity"}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
    }
    mock_chunk_2 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1, mock_chunk_2]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: BaseMessage | None = None
    chunks_list = list(stream)
    # BaseChatModel.stream() adds an extra chunk after the final chunk from _stream
    assert len(chunks_list) == 4
    for i, chunk in enumerate(
        chunks_list[:3]
    ):  # Only check first 3 chunks against mock
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"].get("content", "")
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
    # Process the 4th chunk
    assert full is not None
    full = cast(BaseMessage, full + chunks_list[3])
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {"citations": ["example.com", "example2.com"]}

    patcher.assert_called_once()


def test_perplexity_stream_includes_citations_and_images(mocker: MockerFixture) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)
    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
        "images": [
            {
                "image_url": "mock_image_url",
                "origin_url": "mock_origin_url",
                "height": 100,
                "width": 100,
            }
        ],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {"content": "Perplexity"}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
        "images": [
            {
                "image_url": "mock_image_url",
                "origin_url": "mock_origin_url",
                "height": 100,
                "width": 100,
            }
        ],
    }
    mock_chunk_2 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1, mock_chunk_2]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: BaseMessage | None = None
    chunks_list = list(stream)
    # BaseChatModel.stream() adds an extra chunk after the final chunk from _stream
    assert len(chunks_list) == 4
    for i, chunk in enumerate(
        chunks_list[:3]
    ):  # Only check first 3 chunks against mock
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"].get("content", "")
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
            assert chunk.additional_kwargs["images"] == [
                {
                    "image_url": "mock_image_url",
                    "origin_url": "mock_origin_url",
                    "height": 100,
                    "width": 100,
                }
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
            assert "images" not in chunk.additional_kwargs
    # Process the 4th chunk
    assert full is not None
    full = cast(BaseMessage, full + chunks_list[3])
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {
        "citations": ["example.com", "example2.com"],
        "images": [
            {
                "image_url": "mock_image_url",
                "origin_url": "mock_origin_url",
                "height": 100,
                "width": 100,
            }
        ],
    }

    patcher.assert_called_once()


def test_perplexity_stream_includes_citations_and_related_questions(
    mocker: MockerFixture,
) -> None:
    """Test that the stream method includes citations in the additional_kwargs."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)
    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
        "related_questions": ["example_question_1", "example_question_2"],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {"content": "Perplexity"}, "finish_reason": None}],
        "citations": ["example.com", "example2.com"],
        "related_questions": ["example_question_1", "example_question_2"],
    }
    mock_chunk_2 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1, mock_chunk_2]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: BaseMessage | None = None
    chunks_list = list(stream)
    # BaseChatModel.stream() adds an extra chunk after the final chunk from _stream
    assert len(chunks_list) == 4
    for i, chunk in enumerate(
        chunks_list[:3]
    ):  # Only check first 3 chunks against mock
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"].get("content", "")
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
            assert chunk.additional_kwargs["related_questions"] == [
                "example_question_1",
                "example_question_2",
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
            assert "related_questions" not in chunk.additional_kwargs
    # Process the 4th chunk
    assert full is not None
    full = cast(BaseMessage, full + chunks_list[3])
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {
        "citations": ["example.com", "example2.com"],
        "related_questions": ["example_question_1", "example_question_2"],
    }

    patcher.assert_called_once()


def test_perplexity_stream_includes_citations_and_search_results(
    mocker: MockerFixture,
) -> None:
    """Test that the stream method exposes `search_results` via additional_kwargs."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
        "citations": ["example.com/a", "example.com/b"],
        "search_results": [
            {"title": "Mock result", "url": "https://example.com/result", "date": None}
        ],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {"content": "Perplexity"}, "finish_reason": None}],
        "citations": ["example.com/a", "example.com/b"],
        "search_results": [
            {"title": "Mock result", "url": "https://example.com/result", "date": None}
        ],
    }
    mock_chunk_2 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1, mock_chunk_2]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: BaseMessage | None = None
    chunks_list = list(stream)
    # BaseChatModel.stream() adds an extra chunk after the final chunk from _stream
    assert len(chunks_list) == 4
    for i, chunk in enumerate(
        chunks_list[:3]
    ):  # Only check first 3 chunks against mock
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"].get("content", "")
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com/a",
                "example.com/b",
            ]
            assert chunk.additional_kwargs["search_results"] == [
                {
                    "title": "Mock result",
                    "url": "https://example.com/result",
                    "date": None,
                }
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
            assert "search_results" not in chunk.additional_kwargs
    # Process the 4th chunk
    assert full is not None
    full = cast(BaseMessage, full + chunks_list[3])
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {
        "citations": ["example.com/a", "example.com/b"],
        "search_results": [
            {"title": "Mock result", "url": "https://example.com/result", "date": None}
        ],
    }

    patcher.assert_called_once()


def test_create_usage_metadata_basic() -> None:
    """Test _create_usage_metadata with basic token counts."""
    token_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 10
    assert usage_metadata["output_tokens"] == 20
    assert usage_metadata["total_tokens"] == 30
    assert usage_metadata["output_token_details"]["reasoning"] == 0
    assert usage_metadata["output_token_details"]["citation_tokens"] == 0  # type: ignore[typeddict-item]


def test_create_usage_metadata_with_reasoning_tokens() -> None:
    """Test _create_usage_metadata with reasoning tokens."""
    token_usage = {
        "prompt_tokens": 50,
        "completion_tokens": 100,
        "total_tokens": 150,
        "reasoning_tokens": 25,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 50
    assert usage_metadata["output_tokens"] == 100
    assert usage_metadata["total_tokens"] == 150
    assert usage_metadata["output_token_details"]["reasoning"] == 25
    assert usage_metadata["output_token_details"]["citation_tokens"] == 0  # type: ignore[typeddict-item]


def test_create_usage_metadata_with_citation_tokens() -> None:
    """Test _create_usage_metadata with citation tokens."""
    token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 200,
        "total_tokens": 300,
        "citation_tokens": 15,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 100
    assert usage_metadata["output_tokens"] == 200
    assert usage_metadata["total_tokens"] == 300
    assert usage_metadata["output_token_details"]["reasoning"] == 0
    assert usage_metadata["output_token_details"]["citation_tokens"] == 15  # type: ignore[typeddict-item]


def test_create_usage_metadata_with_all_token_types() -> None:
    """Test _create_usage_metadata with all token types.

    Tests reasoning tokens and citation tokens together.
    """
    token_usage = {
        "prompt_tokens": 75,
        "completion_tokens": 150,
        "total_tokens": 225,
        "reasoning_tokens": 30,
        "citation_tokens": 20,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 75
    assert usage_metadata["output_tokens"] == 150
    assert usage_metadata["total_tokens"] == 225
    assert usage_metadata["output_token_details"]["reasoning"] == 30
    assert usage_metadata["output_token_details"]["citation_tokens"] == 20  # type: ignore[typeddict-item]


def test_create_usage_metadata_missing_optional_fields() -> None:
    """Test _create_usage_metadata with missing optional fields defaults to 0."""
    token_usage = {
        "prompt_tokens": 25,
        "completion_tokens": 50,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 25
    assert usage_metadata["output_tokens"] == 50
    # Total tokens should be calculated if not provided
    assert usage_metadata["total_tokens"] == 75
    assert usage_metadata["output_token_details"]["reasoning"] == 0
    assert usage_metadata["output_token_details"]["citation_tokens"] == 0  # type: ignore[typeddict-item]


def test_create_usage_metadata_empty_dict() -> None:
    """Test _create_usage_metadata with empty token usage dict."""
    token_usage: dict = {}

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 0
    assert usage_metadata["output_tokens"] == 0
    assert usage_metadata["total_tokens"] == 0
    assert usage_metadata["output_token_details"]["reasoning"] == 0
    assert usage_metadata["output_token_details"]["citation_tokens"] == 0  # type: ignore[typeddict-item]


def test_perplexity_invoke_includes_num_search_queries(mocker: MockerFixture) -> None:
    """Test that invoke includes num_search_queries in response_metadata."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    mock_usage = MagicMock()
    mock_usage.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "num_search_queries": 3,
    }

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Test response",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.model = "test-model"
    mock_response.usage = mock_usage

    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    )

    result = llm.invoke("Test query")

    assert result.response_metadata["num_search_queries"] == 3
    assert result.response_metadata["model_name"] == "test-model"
    patcher.assert_called_once()


def test_perplexity_invoke_without_num_search_queries(mocker: MockerFixture) -> None:
    """Test that invoke works when num_search_queries is not provided."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    mock_usage = MagicMock()
    mock_usage.model_dump.return_value = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content="Test response",
                tool_calls=None,
            ),
            finish_reason="stop",
        )
    ]
    mock_response.model = "test-model"
    mock_response.usage = mock_usage

    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    )

    result = llm.invoke("Test query")

    assert "num_search_queries" not in result.response_metadata
    assert result.response_metadata["model_name"] == "test-model"
    patcher.assert_called_once()


def test_perplexity_stream_includes_num_search_queries(mocker: MockerFixture) -> None:
    """Test that stream properly handles num_search_queries in usage data."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Hello "}, "finish_reason": None}],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {"content": "world"}, "finish_reason": None}],
    }
    mock_chunk_2 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 10,
            "total_tokens": 15,
            "num_search_queries": 2,
            "reasoning_tokens": 1,
            "citation_tokens": 3,
        },
    }
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1, mock_chunk_2]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks

    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )

    chunks_list = list(llm.stream("Test query"))

    # Find the chunk with usage metadata
    usage_chunk = None
    for chunk in chunks_list:
        if chunk.usage_metadata:
            usage_chunk = chunk
            break

    # Verify usage metadata is properly set
    assert usage_chunk is not None
    assert usage_chunk.usage_metadata is not None
    assert usage_chunk.usage_metadata["input_tokens"] == 5
    assert usage_chunk.usage_metadata["output_tokens"] == 10
    assert usage_chunk.usage_metadata["total_tokens"] == 15
    # Verify reasoning and citation tokens are included
    assert usage_chunk.usage_metadata["output_token_details"]["reasoning"] == 1
    assert usage_chunk.usage_metadata["output_token_details"]["citation_tokens"] == 3  # type: ignore[typeddict-item]

    patcher.assert_called_once()


def test_profile() -> None:
    model = ChatPerplexity(model="sonar")
    assert model.profile
