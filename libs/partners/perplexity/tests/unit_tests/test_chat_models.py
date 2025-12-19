from typing import Any, cast
from unittest.mock import MagicMock

from langchain_core.messages import AIMessageChunk, BaseMessage
from pytest_mock import MockerFixture

from langchain_perplexity import ChatPerplexity, MediaResponse, WebSearchOptions
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


def test_perplexity_new_params() -> None:
    """Test new Perplexity-specific parameters."""
    web_search_options = WebSearchOptions(search_type="pro", search_context_size="high")
    media_response = MediaResponse(overrides={"return_videos": True})

    llm = ChatPerplexity(
        model="sonar-pro",
        search_mode="academic",
        web_search_options=web_search_options,
        media_response=media_response,
        return_images=True,
    )

    params = llm._default_params
    assert params["search_mode"] == "academic"
    assert params["web_search_options"] == {
        "search_type": "pro",
        "search_context_size": "high",
    }

    assert params["extra_body"]["media_response"] == {
        "overrides": {"return_videos": True}
    }
    assert params["return_images"] is True


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


def test_perplexity_stream_includes_videos_and_reasoning(mocker: MockerFixture) -> None:
    """Test that stream extracts videos and reasoning_steps."""
    llm = ChatPerplexity(model="test", timeout=30, verbose=True)

    mock_chunk_0 = {
        "choices": [{"delta": {"content": "Thinking... "}, "finish_reason": None}],
        "videos": [{"url": "http://video.com", "thumbnail_url": "http://thumb.com"}],
        "reasoning_steps": [{"thought": "I should search", "type": "web_search"}],
    }
    mock_chunk_1 = {
        "choices": [{"delta": {}, "finish_reason": "stop"}],
    }

    mock_chunks: list[dict[str, Any]] = [mock_chunk_0, mock_chunk_1]
    mock_stream = MagicMock()
    mock_stream.__iter__.return_value = mock_chunks
    mocker.patch.object(llm.client.chat.completions, "create", return_value=mock_stream)

    stream = list(llm.stream("test"))
    first_chunk = stream[0]

    assert "videos" in first_chunk.additional_kwargs
    assert first_chunk.additional_kwargs["videos"][0]["url"] == "http://video.com"
    assert "reasoning_steps" in first_chunk.additional_kwargs
    assert (
        first_chunk.additional_kwargs["reasoning_steps"][0]["thought"]
        == "I should search"
    )


def test_create_usage_metadata_basic() -> None:
    """Test _create_usage_metadata with basic token counts."""
    token_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "reasoning_tokens": 0,
        "citation_tokens": 0,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 10
    assert usage_metadata["output_tokens"] == 20
    assert usage_metadata["total_tokens"] == 30
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
        "search_context_size": "high",
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
    # Mock optional fields as empty/None
    mock_response.videos = None
    mock_response.reasoning_steps = None
    mock_response.citations = None
    mock_response.search_results = None
    mock_response.images = None
    mock_response.related_questions = None

    patcher = mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    )

    result = llm.invoke("Test query")

    assert result.response_metadata["num_search_queries"] == 3
    assert result.response_metadata["search_context_size"] == "high"
    assert result.response_metadata["model_name"] == "test-model"
    patcher.assert_called_once()


def test_profile() -> None:
    model = ChatPerplexity(model="sonar")
    assert model.profile
