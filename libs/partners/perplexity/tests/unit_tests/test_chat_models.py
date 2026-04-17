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


# ---------------------------------------------------------------------------
# Responses (Agent) API tests
# ---------------------------------------------------------------------------


def test_use_responses_api_flag() -> None:
    """use_responses_api defaults to False and can be set to True."""
    llm = ChatPerplexity(model="sonar")
    assert llm.use_responses_api is False

    llm_agent = ChatPerplexity(model="openai/gpt-5.4", use_responses_api=True)
    assert llm_agent.use_responses_api is True


def test_convert_messages_to_responses_input_basic() -> None:
    """Human and AI messages are converted to input_items; no instructions."""
    from langchain_core.messages import AIMessage, HumanMessage

    llm = ChatPerplexity(model="openai/gpt-5.4", use_responses_api=True)
    messages = [HumanMessage(content="Hello"), AIMessage(content="Hi there")]
    input_items, instructions = llm._convert_messages_to_responses_input(messages)

    assert instructions is None
    assert len(input_items) == 2
    assert input_items[0] == {"type": "message", "role": "user", "content": "Hello"}
    assert input_items[1] == {
        "type": "message",
        "role": "assistant",
        "content": "Hi there",
    }


def test_convert_messages_to_responses_input_with_system() -> None:
    """System messages become instructions; multiple are newline-joined."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatPerplexity(model="openai/gpt-5.4", use_responses_api=True)
    messages = [
        SystemMessage(content="You are helpful."),
        SystemMessage(content="Be concise."),
        HumanMessage(content="What is 2+2?"),
    ]
    input_items, instructions = llm._convert_messages_to_responses_input(messages)

    assert instructions == "You are helpful.\nBe concise."
    assert len(input_items) == 1
    assert input_items[0]["role"] == "user"


def test_responses_default_params_max_tokens() -> None:
    """max_tokens maps to max_output_tokens in responses params."""
    llm = ChatPerplexity(
        model="openai/gpt-5.4",
        use_responses_api=True,
        max_tokens=512,
        temperature=0.5,
    )
    params = llm._responses_default_params
    assert params["max_output_tokens"] == 512
    assert params["temperature"] == 0.5
    assert "max_tokens" not in params


def test_responses_default_params_no_sonar_fields() -> None:
    """Sonar-specific params (search_mode etc.) must NOT appear in responses params."""
    llm = ChatPerplexity(
        model="openai/gpt-5.4",
        use_responses_api=True,
        search_mode="web",  # type: ignore[arg-type]
    )
    params = llm._responses_default_params
    assert "search_mode" not in params


def test_invoke_with_responses_api(mocker: MockerFixture) -> None:
    """_generate routes to responses.create when use_responses_api=True."""
    llm = ChatPerplexity(model="openai/gpt-5.4", use_responses_api=True)

    mock_response = MagicMock()
    mock_response.output_text = "The answer is 42."
    mock_response.model = "openai/gpt-5.4"
    mock_response.output = []

    mock_usage = MagicMock()
    mock_usage.input_tokens = 10
    mock_usage.output_tokens = 5
    mock_usage.total_tokens = 15
    mock_response.usage = mock_usage

    patcher = mocker.patch.object(
        llm.client.responses, "create", return_value=mock_response
    )

    result = llm.invoke("What is 6 times 7?")

    patcher.assert_called_once()
    call_kwargs = patcher.call_args
    # input must be a list of message dicts
    assert call_kwargs.kwargs["input"][0]["role"] == "user"
    assert result.content == "The answer is 42."
    assert result.usage_metadata["input_tokens"] == 10
    assert result.usage_metadata["output_tokens"] == 5


def test_invoke_with_responses_api_system_message(mocker: MockerFixture) -> None:
    """System messages are passed as instructions, not as input items."""
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = ChatPerplexity(model="openai/gpt-5.4", use_responses_api=True)

    mock_response = MagicMock()
    mock_response.output_text = "Hello!"
    mock_response.model = "openai/gpt-5.4"
    mock_response.output = []
    mock_usage = MagicMock()
    mock_usage.input_tokens = 5
    mock_usage.output_tokens = 3
    mock_usage.total_tokens = 8
    mock_response.usage = mock_usage

    patcher = mocker.patch.object(
        llm.client.responses, "create", return_value=mock_response
    )

    messages = [SystemMessage(content="Be brief."), HumanMessage(content="Hi")]
    llm.invoke(messages)

    call_kwargs = patcher.call_args.kwargs
    assert call_kwargs["instructions"] == "Be brief."
    # Only the human message should appear in input
    assert len(call_kwargs["input"]) == 1
    assert call_kwargs["input"][0]["role"] == "user"


def test_stream_with_responses_api(mocker: MockerFixture) -> None:
    """Streaming with use_responses_api=True uses responses.create(stream=True)."""
    llm = ChatPerplexity(model="openai/gpt-5.4", use_responses_api=True, streaming=True)

    delta_event = MagicMock()
    delta_event.model_dump.return_value = {
        "type": "response.output_text.delta",
        "delta": "Hello ",
    }
    delta_event2 = MagicMock()
    delta_event2.model_dump.return_value = {
        "type": "response.output_text.delta",
        "delta": "world!",
    }
    completed_event = MagicMock()
    completed_event.model_dump.return_value = {
        "type": "response.completed",
        "response": {
            "model": "openai/gpt-5.4",
            "usage": {"input_tokens": 8, "output_tokens": 4, "total_tokens": 12},
        },
    }

    mock_stream = MagicMock()
    mock_stream.__iter__ = MagicMock(
        return_value=iter([delta_event, delta_event2, completed_event])
    )

    patcher = mocker.patch.object(
        llm.client.responses, "create", return_value=mock_stream
    )

    chunks = list(llm.stream("Say hello"))
    patcher.assert_called_once()
    call_kwargs = patcher.call_args.kwargs
    assert call_kwargs.get("stream") is True

    # Collect text from chunks (exclude the final usage chunk)
    text_chunks = [c for c in chunks if c.content]
    full_text = "".join(c.content for c in text_chunks)
    assert "Hello " in full_text
    assert "world!" in full_text
