from typing import Any, Optional, cast
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
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: Optional[BaseMessage] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
        if i == 0:
            assert chunk.additional_kwargs["citations"] == [
                "example.com",
                "example2.com",
            ]
        else:
            assert "citations" not in chunk.additional_kwargs
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
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: Optional[BaseMessage] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
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
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: Optional[BaseMessage] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
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
    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_stream
    )
    stream = llm.stream("Hello langchain")
    full: Optional[BaseMessage] = None
    for i, chunk in enumerate(stream):
        full = chunk if full is None else cast(BaseMessage, full + chunk)
        assert chunk.content == mock_chunks[i]["choices"][0]["delta"]["content"]
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
    assert isinstance(full, AIMessageChunk)
    assert full.content == "Hello Perplexity"
    assert full.additional_kwargs == {
        "citations": ["example.com/a", "example.com/b"],
        "search_results": [
            {"title": "Mock result", "url": "https://example.com/result", "date": None}
        ],
    }

    patcher.assert_called_once()


def test_perplexity_structured_output_preserves_citations(
    mocker: MockerFixture,
) -> None:
    """Test that structured output preserves citations in the response."""
    from pydantic import BaseModel, Field

    class OutputWithCitations(BaseModel):
        """Test model with citations field."""

        result: str = Field(description="The result text")
        citations: list[str] = Field(default_factory=list, description="Citations")

    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    # Mock the API response with proper attributes for citations
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='{"result": "Paris is the capital of France"}')
        )
    ]
    mock_response.model = "test"

    # Set up mock to have citations as an attribute
    mock_response.citations = ["example.com", "example2.com"]

    # Mock hasattr to return True for our mock_response attributes
    def custom_hasattr(obj: Any, name: str) -> bool:
        if obj is mock_response and name == "citations":
            return True
        return (
            object.__getattribute__(obj, "__dict__").get(name) is not None
            if hasattr(obj, "__dict__")
            else False
        )

    with mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    ):
        # Test with model that has citations field
        structured_llm = llm.with_structured_output(OutputWithCitations)
        result = structured_llm.invoke("what is the capital of France?")

        # Verify the result has citations
        assert isinstance(result, OutputWithCitations)
        assert result.result == "Paris is the capital of France"
        assert result.citations == ["example.com", "example2.com"]


def test_perplexity_structured_output_dict_preserves_citations(
    mocker: MockerFixture,
) -> None:
    """Test that structured output with dict schema preserves citations."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='{"result": "Paris is the capital of France"}')
        )
    ]
    mock_response.model = "test"
    mock_response.citations = ["example.com", "example2.com"]

    with mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    ):
        # Test with dict output (JSON mode) - requires title for OpenAI function format
        structured_llm_json = llm.with_structured_output(
            {
                "title": "Output",
                "type": "object",
                "properties": {"result": {"type": "string"}},
            }
        )
        result_json = structured_llm_json.invoke("what is the capital of France?")

        # Verify the result is a dict with citations
        assert isinstance(result_json, dict)
        assert result_json["result"] == "Paris is the capital of France"
        assert result_json.get("citations") == ["example.com", "example2.com"]


def test_perplexity_structured_output_without_citations_field(
    mocker: MockerFixture,
) -> None:
    """Test that structured output works when model doesn't have citations field."""
    from pydantic import BaseModel, Field

    class SimpleOutput(BaseModel):
        """Test model without citations field."""

        result: str = Field(description="The result text")

    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    # Mock the API response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content='{"result": "Paris is the capital of France"}')
        )
    ]
    mock_response.model = "test"
    mock_response.citations = ["example.com", "example2.com"]

    with mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    ):
        # Test with model that doesn't have citations field
        structured_llm = llm.with_structured_output(SimpleOutput)
        result = structured_llm.invoke("what is the capital of France?")

        # Verify the result doesn't have citations (model doesn't have field)
        assert isinstance(result, SimpleOutput)
        assert result.result == "Paris is the capital of France"
        assert not hasattr(result, "citations")
