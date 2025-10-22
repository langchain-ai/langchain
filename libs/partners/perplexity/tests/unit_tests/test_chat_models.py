from typing import Any, cast
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, BaseMessage
from pytest_mock import MockerFixture

from langchain_perplexity import ChatPerplexity


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
