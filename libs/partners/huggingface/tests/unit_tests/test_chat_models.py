from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatResult
from langchain_core.tools import BaseTool

from langchain_huggingface.chat_models import (  # type: ignore[import]
    ChatHuggingFace,
    _convert_dict_to_message,
)
from langchain_huggingface.llms import HuggingFaceEndpoint, HuggingFacePipeline


@pytest.fixture
def mock_hf_endpoint_llm() -> Mock:
    llm = Mock(spec=HuggingFaceEndpoint)
    llm.inference_server_url = "test endpoint url"
    # Mock the __dict__ to simulate presence of attributes for kwarg filtering
    llm.__dict__ = {"some_llm_param": True}
    return llm


@pytest.fixture
def mock_hf_pipeline_llm() -> Mock:
    llm = Mock(spec=HuggingFacePipeline)
    llm.model_id = "test-model-id"
    # Mock the pipeline attribute expected by _stream and _generate
    llm.pipeline = MagicMock()
    llm.pipeline.return_value = "test output"  # for invoke

    # For stream, the pipeline should return an iterator
    mock_stream_pipeline = MagicMock()
    mock_stream_pipeline.return_value = iter(["test ", "output"])
    llm._stream = MagicMock(
        side_effect=lambda prompt, **kwargs: iter([MagicMock(text=prompt)])
    )  # Simplified mock

    # Mock the __dict__ to simulate presence of attributes for kwarg filtering
    llm.__dict__ = {"some_llm_param": True}
    return llm


@pytest.fixture
@patch("langchain_huggingface.chat_models.huggingface.AutoTokenizer.from_pretrained")
def chat_hugging_face_pipeline(
    mock_from_pretrained: MagicMock, mock_hf_pipeline_llm: Mock
) -> ChatHuggingFace:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value="formatted_prompt")
    mock_from_pretrained.return_value = mock_tokenizer
    # Initialize with a pipeline LLM
    chat_hf = ChatHuggingFace(llm=mock_hf_pipeline_llm, tokenizer=mock_tokenizer)
    # chat_hf._resolve_model_id() # Ensure tokenizer is resolved
    return chat_hf


@pytest.fixture
@patch(
    "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
)  # Keep this for tests that might use the endpoint based mock_llm
def chat_hugging_face_endpoint(
    mock_resolve_id: Any, mock_hf_endpoint_llm: Any
) -> ChatHuggingFace:
    chat_hf = ChatHuggingFace(llm=mock_hf_endpoint_llm, tokenizer=MagicMock())
    return chat_hf


def test_create_chat_result(chat_hugging_face_endpoint: Any) -> None:
    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "test message"},
                "finish_reason": "test finish reason",
            }
        ],
        "usage": {"tokens": 420},
    }

    result = chat_hugging_face_endpoint._create_chat_result(mock_response)
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "test message"
    assert (
        result.generations[0].generation_info["finish_reason"] == "test finish reason"  # type: ignore[index]
    )
    assert result.llm_output["token_usage"]["tokens"] == 420  # type: ignore[index]
    assert result.llm_output["model_name"] == chat_hugging_face_endpoint.model_id  # type: ignore[index]


@pytest.mark.parametrize(
    "messages, expected_error",
    [
        ([], "At least one HumanMessage must be provided!"),
        (
            [HumanMessage(content="Hi"), AIMessage(content="Hello")],
            "Last message must be a HumanMessage!",
        ),
    ],
)
def test_to_chat_prompt_errors(
    chat_hugging_face_pipeline: Any, messages: list[BaseMessage], expected_error: str
) -> None:
    with pytest.raises(ValueError) as e:
        chat_hugging_face_pipeline._to_chat_prompt(messages)
    assert expected_error in str(e.value)


def test_to_chat_prompt_valid_messages_no_kwargs(
    chat_hugging_face_pipeline: Any,
) -> None:
    messages = [AIMessage(content="Hello"), HumanMessage(content="How are you?")]
    expected_prompt = "Generated chat prompt"

    chat_hugging_face_pipeline.tokenizer.apply_chat_template.return_value = (
        expected_prompt
    )

    result = chat_hugging_face_pipeline._to_chat_prompt(messages)

    assert result == expected_prompt
    chat_hugging_face_pipeline.tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


def test_to_chat_prompt_with_extra_kwargs(chat_hugging_face_pipeline: Any) -> None:
    messages = [AIMessage(content="Hello"), HumanMessage(content="How are you?")]
    expected_prompt = "Generated chat prompt with kwargs"
    chat_hugging_face_pipeline.tokenizer.apply_chat_template.return_value = (
        expected_prompt
    )
    extra_kwargs = {"foo": "bar", "another_arg": 123}

    result = chat_hugging_face_pipeline._to_chat_prompt(messages, **extra_kwargs)

    assert result == expected_prompt
    chat_hugging_face_pipeline.tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
        **extra_kwargs,
    )


@patch("langchain_huggingface.chat_models.huggingface.AutoTokenizer.from_pretrained")
def test_custom_chat_template_for_pipeline(
    mock_from_pretrained: MagicMock, mock_hf_pipeline_llm: Mock
) -> None:
    custom_template = (
        "{% for message in messages %}"
        "{{ message.role }}: {{ message.content }}"
        "{% endfor %}"
    )
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock()
    mock_from_pretrained.return_value = mock_tokenizer

    # Need to re-initialize to trigger _resolve_model_id via model_validator
    chat = ChatHuggingFace(llm=mock_hf_pipeline_llm, chat_template=custom_template)

    assert chat.tokenizer is not None
    assert chat.tokenizer.chat_template == custom_template

    # Verify it's used in _to_chat_prompt
    messages = [HumanMessage(content="Test")]
    chat._to_chat_prompt(messages)
    chat.tokenizer.apply_chat_template.assert_called_once()


@patch("langchain_huggingface.chat_models.huggingface.AutoTokenizer.from_pretrained")
def test_invoke_with_pipeline_passes_kwargs_to_template(
    mock_from_pretrained: MagicMock, mock_hf_pipeline_llm: Mock
) -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(
        return_value="formatted_prompt_for_invoke"
    )
    mock_from_pretrained.return_value = mock_tokenizer

    chat = ChatHuggingFace(llm=mock_hf_pipeline_llm, tokenizer=mock_tokenizer)

    # Mock the llm's generate method as it's called by invoke
    chat.llm._generate = MagicMock(
        return_value=ChatResult(
            generations=[MagicMock(message=AIMessage(content="response"))]
        )
    )

    messages = [HumanMessage(content="Hi")]
    template_kwargs = {"custom_param": "value1", "another_one": False}

    # We also pass some llm specific kwargs to ensure they are filtered if necessary
    llm_params = {"max_new_tokens": 100}
    all_kwargs = {**template_kwargs, **llm_params}

    chat.invoke(messages, **all_kwargs)

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "Hi"}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,  # Assert that only template kwargs are passed
    )
    # Assert that llm._generate was called with its specific params.
    # The current filtering (`k in self.llm.__dict__`) is a basic heuristic.
    # It assumes that kwargs not in llm.__dict__ were intended for the template.
    # For HuggingFacePipeline, _generate receives llm_specific_kwargs.
    # For example, if max_new_tokens is not in llm.__dict__,
    # it would be filtered out before calling llm._generate.
    # Refinement of this logic may be needed for more complex scenarios.


@patch("langchain_huggingface.chat_models.huggingface.AutoTokenizer.from_pretrained")
def test_stream_with_pipeline_passes_kwargs_to_template(
    mock_from_pretrained: MagicMock, mock_hf_pipeline_llm: Mock
) -> None:
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(
        return_value="formatted_prompt_for_stream"
    )
    mock_from_pretrained.return_value = mock_tokenizer

    chat = ChatHuggingFace(llm=mock_hf_pipeline_llm, tokenizer=mock_tokenizer)

    # Mock the llm's stream method
    # chat.llm._stream = MagicMock(return_value=iter([MagicMock(text="response")]))

    messages = [HumanMessage(content="Hi stream")]
    template_kwargs = {"stream_param": True, "detail_level": "high"}
    llm_params = {"temperature": 0.5}  # Example LLM param
    all_kwargs = {**template_kwargs, **llm_params}

    list(chat.stream(messages, **all_kwargs))

    mock_tokenizer.apply_chat_template.assert_called_once_with(
        [{"role": "user", "content": "Hi stream"}],
        tokenize=False,
        add_generation_prompt=True,
        **template_kwargs,  # As above, check how kwargs are filtered
    )
    # chat.llm._stream.assert_called_once()
    # Check if it was called, args are harder to assert precisely here


@pytest.mark.asyncio
@patch("langchain_huggingface.chat_models.huggingface.AutoTokenizer.from_pretrained")
async def test_ainvoke_with_endpoint_passes_kwargs(
    mock_from_pretrained: MagicMock, mock_hf_endpoint_llm: Mock
) -> None:
    # This test uses HuggingFaceEndpoint as pipeline doesn't support async well
    mock_tokenizer = MagicMock()
    mock_tokenizer.apply_chat_template = MagicMock(return_value="formatted_prompt")
    mock_from_pretrained.return_value = mock_tokenizer

    # Mock the async client for HuggingFaceEndpoint
    mock_async_client = MagicMock()
    mock_async_client.chat_completion = AsyncMock(
        return_value={
            "choices": [
                {"message": {"role": "assistant", "content": "async response"}}
            ],
            "usage": {},
        }
    )
    mock_hf_endpoint_llm.async_client = mock_async_client

    # Re-initialize ChatHuggingFace with the mocked tokenizer
    # if _resolve_model_id is also mocked for endpoint.
    # For this test, let's assume tokenizer is correctly set up.
    chat = ChatHuggingFace(llm=mock_hf_endpoint_llm, tokenizer=mock_tokenizer)

    messages = [HumanMessage(content="Hi async")]
    # Kwargs for HuggingFaceEndpoint are passed directly to its client
    # So, apply_chat_template (if it were called for endpoint) wouldn't get these.
    # This test is more about ensuring the flow for endpoint.
    # The _to_chat_prompt is NOT called for Endpoint in _agenerate.
    # So we test that chat_completion is called with the kwargs.

    endpoint_kwargs = {"max_tokens": 50, "temperature": 0.7}
    await chat.ainvoke(messages, **endpoint_kwargs)

    # Assert that the endpoint's async_client.chat_completion
    # was called with these kwargs.
    # _create_message_dicts is called first, then chat_completion.
    # The kwargs are passed to chat_completion.
    expected_call_kwargs = {
        **chat.model_kwargs,
        **endpoint_kwargs,
    }  # model_kwargs is {} by default

    # We need to check the second argument of the call (kwargs)
    # call_args[0] is a tuple of positional args,
    # call_args[1] is a dict of kwargs
    args, called_kwargs = mock_async_client.chat_completion.call_args
    assert "messages" in called_kwargs  # messages is a kwarg here
    for k, v in expected_call_kwargs.items():
        assert k in called_kwargs and called_kwargs[k] == v


@pytest.mark.asyncio
@patch("langchain_huggingface.chat_models.huggingface.AutoTokenizer.from_pretrained")
async def test_astream_with_endpoint_passes_kwargs(
    mock_from_pretrained: MagicMock, mock_hf_endpoint_llm: Mock
) -> None:
    mock_tokenizer = (
        MagicMock()
    )  # Not directly used by endpoint logic for apply_chat_template
    mock_from_pretrained.return_value = mock_tokenizer

    mock_async_client = MagicMock()

    # Mock async generator for chat_completion
    async def mock_gen(*args: Any, **kwargs: Any) -> AsyncIterator[dict]:
        yield {
            "choices": [{"delta": {"role": "assistant", "content": "async stream"}}],
            "usage": {},
        }

    mock_async_client.chat_completion = MagicMock(
        side_effect=mock_gen
    )  # Use MagicMock for side_effect
    mock_hf_endpoint_llm.async_client = mock_async_client

    chat = ChatHuggingFace(llm=mock_hf_endpoint_llm, tokenizer=mock_tokenizer)

    messages = [HumanMessage(content="Hi async stream")]
    endpoint_kwargs = {"top_p": 0.9, "details": True}  # Example endpoint kwargs

    results = []
    async for chunk in chat.astream(messages, **endpoint_kwargs):
        results.append(chunk)

    assert len(results) > 0

    expected_call_kwargs = {**chat.model_kwargs, **endpoint_kwargs, "stream": True}
    args, called_kwargs = mock_async_client.chat_completion.call_args
    assert "messages" in called_kwargs
    for k, v in expected_call_kwargs.items():
        assert k in called_kwargs and called_kwargs[k] == v


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        (
            SystemMessage(content="You are a helpful assistant."),
            {"role": "system", "content": "You are a helpful assistant."},
        ),
        (
            AIMessage(content="How can I help you?"),
            {"role": "assistant", "content": "How can I help you?"},
        ),
        (
            HumanMessage(content="Hello"),
            {"role": "user", "content": "Hello"},
        ),
    ],
)
def test_to_chatml_format(
    chat_hugging_face_pipeline: Any, message: BaseMessage, expected: dict[str, str]
) -> None:
    result = chat_hugging_face_pipeline._to_chatml_format(message)
    assert result == expected


def test_to_chatml_format_with_invalid_type(chat_hugging_face_pipeline: Any) -> None:
    message = FunctionMessage(
        name="func", content="invalid"
    )  # Use a concrete BaseMessage subclass
    with pytest.raises(ValueError) as e:
        chat_hugging_face_pipeline._to_chatml_format(message)
    assert "Unknown message type:" in str(e.value)


@pytest.mark.parametrize(
    ("msg_dict", "expected_type", "expected_content"),
    [
        (
            {"role": "system", "content": "You are helpful"},
            SystemMessage,
            "You are helpful",
        ),
        (
            {"role": "user", "content": "Hello there"},
            HumanMessage,
            "Hello there",
        ),
        (
            {"role": "assistant", "content": "How can I help?"},
            AIMessage,
            "How can I help?",
        ),
        (
            {"role": "function", "content": "result", "name": "get_time"},
            FunctionMessage,
            "result",
        ),
    ],
)
def test_convert_dict_to_message(
    msg_dict: dict[str, Any], expected_type: type, expected_content: str
) -> None:
    result = _convert_dict_to_message(msg_dict)
    assert isinstance(result, expected_type)
    assert result.content == expected_content


# Need to import AsyncMock for Python < 3.8
try:
    from unittest.mock import AsyncMock
except ImportError:
    # Fallback for Python 3.7 if necessary, though tests might need adjustment
    # For this context, assuming Python 3.8+ where AsyncMock is standard
    pass


def tool_mock() -> dict:
    return {"function": {"name": "test_tool"}}


@pytest.mark.parametrize(
    "tools, tool_choice, expected_exception, expected_message",
    [
        ([tool_mock()], ["invalid type"], ValueError, "Unrecognized tool_choice type."),
        (
            [tool_mock(), tool_mock()],
            "test_tool",
            ValueError,
            "must provide exactly one tool.",
        ),
        (
            [tool_mock()],
            {"type": "function", "function": {"name": "other_tool"}},
            ValueError,
            "Tool choice {'type': 'function', 'function': {'name': 'other_tool'}} "
            "was specified, but the only provided tool was test_tool.",
        ),
    ],
)
def test_bind_tools_errors(
    chat_hugging_face: Any,
    tools: dict[str, str],
    tool_choice: Any,
    expected_exception: Any,
    expected_message: str,
) -> None:
    with patch(
        "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
        side_effect=lambda x: x,
    ):
        with pytest.raises(expected_exception) as excinfo:
            chat_hugging_face.bind_tools(tools, tool_choice=tool_choice)
        assert expected_message in str(excinfo.value)


def test_bind_tools(chat_hugging_face: Any) -> None:
    tools = [MagicMock(spec=BaseTool)]
    with (
        patch(
            "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
            side_effect=lambda x: x,
        ),
        patch("langchain_core.runnables.base.Runnable.bind") as mock_super_bind,
    ):
        chat_hugging_face.bind_tools(tools, tool_choice="auto")
        mock_super_bind.assert_called_once()
        _, kwargs = mock_super_bind.call_args
        assert kwargs["tools"] == tools
        assert kwargs["tool_choice"] == "auto"
