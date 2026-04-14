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
    assert "input_token_details" not in usage_metadata


def test_create_usage_metadata_with_search_queries() -> None:
    """Test _create_usage_metadata includes num_search_queries in input_token_details.

    num_search_queries drives per-query billing and must be surfaced in
    input_token_details so that LangSmith can compute accurate call costs.
    """
    token_usage = {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
        "num_search_queries": 3,
        "reasoning_tokens": 5,
        "citation_tokens": 15,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    assert usage_metadata["input_tokens"] == 10
    assert usage_metadata["output_tokens"] == 20
    assert usage_metadata["total_tokens"] == 30
    # num_search_queries drives per-query billing and must be in input_token_details
    assert usage_metadata["input_token_details"]["num_search_queries"] == 3  # type: ignore[typeddict-item]
    assert usage_metadata["output_token_details"]["reasoning"] == 5
    assert usage_metadata["output_token_details"]["citation_tokens"] == 15  # type: ignore[typeddict-item]


def test_create_usage_metadata_zero_search_queries() -> None:
    """Test _create_usage_metadata with num_search_queries=0."""
    token_usage = {
        "prompt_tokens": 5,
        "completion_tokens": 10,
        "total_tokens": 15,
        "num_search_queries": 0,
    }

    usage_metadata = _create_usage_metadata(token_usage)

    # Zero is a valid value — still included so callers can distinguish 0 from absent
    assert usage_metadata["input_token_details"]["num_search_queries"] == 0  # type: ignore[typeddict-item]


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


def test_num_search_queries_flows_into_usage_metadata(
    mocker: MockerFixture,
) -> None:
    """Smoke test: num_search_queries from the API response reaches usage_metadata.

    Verifies the full path: API response -> _create_usage_metadata ->
    AIMessage.usage_metadata.input_token_details["num_search_queries"].
    Previously num_search_queries was only stored in response_metadata, making
    it invisible to LangSmith's cost calculator.
    """
    llm = ChatPerplexity(model="sonar-pro", timeout=30)

    mock_usage = MagicMock()
    mock_usage.model_dump.return_value = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "num_search_queries": 3,
        "citation_tokens": 20,
        "reasoning_tokens": 0,
        "cost": {
            "input_tokens_cost": 0.0001,
            "output_tokens_cost": 0.00005,
            "total_cost": 0.03015,
            "citation_tokens_cost": 0.00002,
            "reasoning_tokens_cost": None,
            "request_cost": None,
            "search_queries_cost": 0.015,
        },
    }

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Paris is the capital.", tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_response.model = "sonar-pro"
    mock_response.usage = mock_usage
    mock_response.citations = None
    mock_response.images = None
    mock_response.related_questions = None
    mock_response.search_results = None
    mock_response.videos = None
    mock_response.reasoning_steps = None

    mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    )

    result = llm.invoke("What is the capital of France?")

    assert result.usage_metadata is not None
    # num_search_queries must be in input_token_details for LangSmith cost tracking
    assert "input_token_details" in result.usage_metadata
    assert result.usage_metadata["input_token_details"]["num_search_queries"] == 3  # type: ignore[typeddict-item]
    # citation_tokens and reasoning must still be in output_token_details
    assert result.usage_metadata["output_token_details"]["citation_tokens"] == 20  # type: ignore[typeddict-item]
    assert result.usage_metadata["output_token_details"]["reasoning"] == 0
    # token counts correct
    assert result.usage_metadata["input_tokens"] == 100
    assert result.usage_metadata["output_tokens"] == 50
    assert result.usage_metadata["total_tokens"] == 150
    # The API-returned cost breakdown must be in response_metadata so callers
    # can read the exact cost Perplexity charged (incl. search_queries_cost)
    assert result.response_metadata["cost"]["total_cost"] == 0.03015
    assert result.response_metadata["cost"]["search_queries_cost"] == 0.015


def test_cost_absent_when_reasoning_mode_enabled(mocker: MockerFixture) -> None:
    """Smoke test: missing cost field (e.g. reasoning mode) does not raise.

    Some providers (e.g. OpenRouter) omit the cost object from the usage
    payload when reasoning mode is enabled. The integration must degrade
    gracefully — no KeyError, no crash — and simply omit 'cost' from
    response_metadata.
    """
    llm = ChatPerplexity(model="sonar-reasoning-pro", timeout=30)

    mock_usage = MagicMock()
    mock_usage.model_dump.return_value = {
        "prompt_tokens": 200,
        "completion_tokens": 80,
        "total_tokens": 280,
        "reasoning_tokens": 60,
        "num_search_queries": 2,
        # cost is absent — simulates reasoning-mode responses from some providers
    }

    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(content="Reasoned answer.", tool_calls=None),
            finish_reason="stop",
        )
    ]
    mock_response.model = "sonar-reasoning-pro"
    mock_response.usage = mock_usage
    mock_response.citations = None
    mock_response.images = None
    mock_response.related_questions = None
    mock_response.search_results = None
    mock_response.videos = None
    mock_response.reasoning_steps = None

    mocker.patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    )

    result = llm.invoke("Explain quantum entanglement.")

    # Must not raise — cost simply absent from response_metadata
    assert "cost" not in result.response_metadata
    # Other fields still correct
    assert result.usage_metadata is not None
    assert result.usage_metadata["input_token_details"]["num_search_queries"] == 2  # type: ignore[typeddict-item]
    assert result.usage_metadata["output_token_details"]["reasoning"] == 60


def test_profile() -> None:
    model = ChatPerplexity(model="sonar")
    assert model.profile
