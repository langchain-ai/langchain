import json
from collections.abc import AsyncIterator, Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest  # type: ignore[import-not-found]
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import RunnableBinding, RunnableSequence
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_json_schema
from pydantic import BaseModel, Field
from pydantic.v1 import BaseModel as BaseModelV1
from pydantic.v1 import Field as FieldV1
from typing_extensions import TypedDict

from langchain_huggingface.chat_models import (  # type: ignore[import]
    ChatHuggingFace,
    _convert_dict_to_message,
)
from langchain_huggingface.llms import HuggingFaceEndpoint


@pytest.fixture
def mock_llm() -> Mock:
    llm = Mock(spec=HuggingFaceEndpoint)
    llm.inference_server_url = "test endpoint url"
    llm.temperature = 0.7
    llm.max_new_tokens = 512
    llm.top_p = 0.9
    llm.seed = 42
    llm.streaming = True
    llm.repetition_penalty = 1.1
    llm.stop_sequences = ["</s>", "<|end|>"]
    llm.model_kwargs = {"do_sample": True, "top_k": 50}
    llm.server_kwargs = {"timeout": 120}
    llm.repo_id = "test/model"
    llm.model = "test/model"
    return llm


@pytest.fixture
@patch(
    "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
)
def chat_hugging_face(mock_resolve_id: Any, mock_llm: Any) -> ChatHuggingFace:
    return ChatHuggingFace(llm=mock_llm, tokenizer=MagicMock())


def test_create_chat_result(chat_hugging_face: Any) -> None:
    mock_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "test message"},
                "finish_reason": "test finish reason",
            }
        ],
        "usage": {"tokens": 420},
    }

    result = chat_hugging_face._create_chat_result(mock_response)
    assert isinstance(result, ChatResult)
    assert result.generations[0].message.content == "test message"
    assert (
        result.generations[0].generation_info["finish_reason"] == "test finish reason"  # type: ignore[index]
    )
    assert result.llm_output["token_usage"]["tokens"] == 420  # type: ignore[index]
    assert result.llm_output["model_name"] == chat_hugging_face.model_id  # type: ignore[index]


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
    chat_hugging_face: Any, messages: list[BaseMessage], expected_error: str
) -> None:
    with pytest.raises(ValueError) as e:
        chat_hugging_face._to_chat_prompt(messages)
    assert expected_error in str(e.value)


def test_to_chat_prompt_valid_messages(chat_hugging_face: Any) -> None:
    messages = [AIMessage(content="Hello"), HumanMessage(content="How are you?")]
    expected_prompt = "Generated chat prompt"

    chat_hugging_face.tokenizer.apply_chat_template.return_value = expected_prompt

    result = chat_hugging_face._to_chat_prompt(messages)

    assert result == expected_prompt
    chat_hugging_face.tokenizer.apply_chat_template.assert_called_once_with(
        [
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )


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
    chat_hugging_face: Any, message: BaseMessage, expected: dict[str, str]
) -> None:
    result = chat_hugging_face._to_chatml_format(message)
    assert result == expected


def test_to_chatml_format_with_invalid_type(chat_hugging_face: Any) -> None:
    message = "Invalid message type"
    with pytest.raises(ValueError) as e:
        chat_hugging_face._to_chatml_format(message)
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


@pytest.mark.parametrize(
    "tool_choice", ["auto", "none", "required", "any", "test_tool"]
)
def test_bind_tools(chat_hugging_face: Any, tool_choice: str) -> None:
    tools = [MagicMock(spec=BaseTool)]
    with (
        patch(
            "langchain_huggingface.chat_models.huggingface.convert_to_openai_tool",
            side_effect=lambda x: x,
        ),
        patch(
            "langchain_core.language_models.chat_models.BaseChatModel.bind"
        ) as mock_super_bind,
    ):
        chat_hugging_face.bind_tools(tools, tool_choice=tool_choice)
        mock_super_bind.assert_called_once()
        _, kwargs = mock_super_bind.call_args
        assert kwargs["tools"] == tools
        expected: Any = tool_choice
        if tool_choice == "any":
            expected = "required"
        elif tool_choice == "test_tool":
            expected = {"type": "function", "function": {"name": "test_tool"}}
        assert kwargs["tool_choice"] == expected


@pytest.mark.parametrize("mode", ["sync", "async"])
@pytest.mark.parametrize("usage_location", ["none", "final", "content", "cumulative"])
async def test_stream_usage_reported_once(
    chat_hugging_face: Any, mode: str, usage_location: str
) -> None:
    usage = {"prompt_tokens": 82, "completion_tokens": 3, "total_tokens": 85}
    responses: list[dict[str, Any]] = [
        {"choices": [{"delta": {"content": text}, "finish_reason": None}]}
        for text in ["Hel", "lo"]
    ]
    if usage_location == "content":
        responses[-1]["usage"] = usage
    elif usage_location in ("final", "cumulative"):
        if usage_location == "cumulative":
            responses[0]["usage"] = {
                "prompt_tokens": 82,
                "completion_tokens": 1,
                "total_tokens": 83,
            }
            responses[-1]["usage"] = usage
        responses.append({"choices": [], "usage": usage})

    async def async_responses() -> AsyncIterator[dict[str, Any]]:
        for response in responses:
            yield response

    chat_hugging_face.llm.client = Mock()
    chat_hugging_face.llm.client.chat_completion.return_value = iter(responses)
    chat_hugging_face.llm.async_client = Mock()
    chat_hugging_face.llm.async_client.chat_completion = AsyncMock(
        return_value=async_responses()
    )
    messages = [HumanMessage("Hello")]
    if mode == "sync":
        chunks = list(chat_hugging_face._stream(messages, stream_usage=True))
    else:
        chunks = [
            chunk
            async for chunk in chat_hugging_face._astream(messages, stream_usage=True)
        ]
    assert "".join(chunk.text for chunk in chunks) == "Hello"
    reported_usage = [
        chunk.message.usage_metadata for chunk in chunks if chunk.message.usage_metadata
    ]
    expected = {"input_tokens": 82, "output_tokens": 3, "total_tokens": 85}
    assert reported_usage == ([] if usage_location == "none" else [expected])
    if reported_usage:
        assert chunks[-1].message.usage_metadata == expected


def test_property_inheritance_integration(chat_hugging_face: Any) -> None:
    """Test that ChatHuggingFace inherits params from LLM object."""
    assert getattr(chat_hugging_face, "temperature", None) == 0.7
    assert getattr(chat_hugging_face, "max_tokens", None) == 512
    assert getattr(chat_hugging_face, "top_p", None) == 0.9
    assert getattr(chat_hugging_face, "streaming", None) is True


def test_default_params_includes_inherited_values(chat_hugging_face: Any) -> None:
    """Test that _default_params includes inherited max_tokens from max_new_tokens."""
    params = chat_hugging_face._default_params
    assert params["max_tokens"] == 512  # inherited from LLM's max_new_tokens
    assert params["temperature"] == 0.7  # inherited from LLM's temperature
    assert params["stream"] is True  # inherited from LLM's streaming


def test_create_message_dicts_includes_inherited_params(chat_hugging_face: Any) -> None:
    """Test that _create_message_dicts includes inherited parameters in API call."""
    messages = [HumanMessage(content="test message")]
    message_dicts, params = chat_hugging_face._create_message_dicts(messages, None)

    # Verify inherited parameters are included
    assert params["max_tokens"] == 512
    assert params["temperature"] == 0.7
    assert params["stream"] is True

    # Verify message conversion
    assert len(message_dicts) == 1
    assert message_dicts[0]["role"] == "user"
    assert message_dicts[0]["content"] == "test message"


def test_model_kwargs_inheritance(mock_llm: Any) -> None:
    """Test that model_kwargs are inherited when not explicitly set."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        assert chat.model_kwargs == {"do_sample": True, "top_k": 50}


def test_huggingface_endpoint_specific_inheritance(mock_llm: Any) -> None:
    """Test HuggingFaceEndpoint specific parameter inheritance."""
    with (
        patch(
            "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
        ),
        patch(
            "langchain_huggingface.chat_models.huggingface._is_huggingface_endpoint",
            return_value=True,
        ),
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        assert (
            getattr(chat, "frequency_penalty", None) == 1.1
        )  # from repetition_penalty


def test_parameter_precedence_explicit_over_inherited(mock_llm: Any) -> None:
    """Test that explicitly set parameters take precedence over inherited ones."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        # Explicitly set max_tokens to override inheritance
        chat = ChatHuggingFace(llm=mock_llm, max_tokens=256, temperature=0.5)
        assert chat.max_tokens == 256  # explicit value, not inherited 512
        assert chat.temperature == 0.5  # explicit value, not inherited 0.7


def test_inheritance_with_no_llm_properties(mock_llm: Any) -> None:
    """Test inheritance when LLM doesn't have expected properties."""
    # Remove some properties from mock
    del mock_llm.temperature
    del mock_llm.top_p

    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        chat = ChatHuggingFace(llm=mock_llm)
        # Should still inherit available properties
        assert chat.max_tokens == 512  # max_new_tokens still available
        # Missing properties should remain None/default
        assert getattr(chat, "temperature", None) is None
        assert getattr(chat, "top_p", None) is None


def test_inheritance_with_empty_llm() -> None:
    """Test that inheritance handles LLM with no relevant attributes gracefully."""
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        # Create a minimal mock LLM that passes validation but has no
        # inheritance attributes
        empty_llm = Mock(spec=HuggingFaceEndpoint)
        empty_llm.repo_id = "test/model"
        empty_llm.model = "test/model"
        # Mock doesn't have the inheritance attributes by default

        chat = ChatHuggingFace(llm=empty_llm)
        # Properties should remain at their default values when LLM has no
        # relevant attrs
        assert chat.max_tokens is None
        assert chat.temperature is None


def test_metadata_versions(chat_hugging_face: Any) -> None:
    """Test that metadata reports the correct version info."""
    from langchain_huggingface._version import __version__

    assert chat_hugging_face.metadata is not None
    versions = chat_hugging_face.metadata["lc_versions"]
    assert "langchain-core" in versions
    assert "langchain-huggingface" in versions
    assert versions["langchain-huggingface"] == __version__


def test_profile() -> None:
    empty_llm = Mock(spec=HuggingFaceEndpoint)
    empty_llm.repo_id = "test/model"
    empty_llm.model = "test/model"

    model = ChatHuggingFace(
        model_id="moonshotai/Kimi-K2-Instruct-0905",
        llm=empty_llm,
    )
    assert model.profile


def test_init_chat_model_huggingface() -> None:
    """Test that init_chat_model works with HuggingFace models.

    This test verifies that the fix for issue #28226 works correctly.
    The issue was that init_chat_model didn't properly handle HuggingFace
    model initialization, particularly the required 'task' parameter and
    parameter separation between HuggingFacePipeline and ChatHuggingFace.
    """
    from langchain.chat_models.base import init_chat_model

    # Test basic initialization with default task
    # Note: This test may skip in CI if model download fails, but it verifies
    # that the initialization code path works correctly
    try:
        llm = init_chat_model(
            model="microsoft/Phi-3-mini-4k-instruct",
            model_provider="huggingface",
            temperature=0,
            max_tokens=1024,
        )

        # Verify that ChatHuggingFace was created successfully
        assert llm is not None
        from langchain_huggingface import ChatHuggingFace

        assert isinstance(llm, ChatHuggingFace)

        # Verify that the llm attribute is set (this was the bug - it was missing)
        assert hasattr(llm, "llm")
        assert llm.llm is not None

        # Test with explicit task parameter
        llm2 = init_chat_model(
            model="microsoft/Phi-3-mini-4k-instruct",
            model_provider="huggingface",
            task="text-generation",
            temperature=0.5,
        )
        assert isinstance(llm2, ChatHuggingFace)
        assert llm2.llm is not None
    except (
        ImportError,
        OSError,
        RuntimeError,
        ValueError,
    ) as e:
        # If model download fails in CI, skip the test rather than failing
        # The important part is that the code path doesn't raise ValidationError
        # about missing 'llm' field, which was the original bug
        pytest.skip(f"Skipping test due to model download/initialization error: {e}")


class _Joke(BaseModel):
    """Joke to tell user."""

    setup: str = Field(description="question to set up a joke")
    punchline: str = Field(description="answer to resolve the joke")


class _JokeV1(BaseModelV1):
    """Joke to tell user."""

    setup: str = FieldV1(description="question to set up a joke")
    punchline: str = FieldV1(description="answer to resolve the joke")


class _OptionalJoke(BaseModel):
    """Joke with an optional punchline."""

    setup: str = Field(description="question to set up a joke")
    punchline: str | None = Field(
        default=None, description="answer to resolve the joke"
    )


class _JokeDict(TypedDict):
    """Joke to tell user."""

    setup: str
    punchline: str


class _StubChatHuggingFace(ChatHuggingFace):
    """Chat model that returns a canned response instead of calling a backend."""

    stub_response: AIMessage

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(
            generations=[ChatGeneration(message=self.stub_response)],
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        stream: bool | None = None,  # noqa: FBT001
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if self.stub_response.tool_calls:
            for index, tool_call in enumerate(self.stub_response.tool_calls):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        tool_call_chunks=[
                            {
                                "name": tool_call["name"],
                                "args": json.dumps(tool_call["args"]),
                                "id": tool_call.get("id"),
                                "index": index,
                            }
                        ],
                    )
                )
        else:
            yield ChatGenerationChunk(
                message=AIMessageChunk(content=self.stub_response.content)
            )


def _stub_chat(mock_llm: Mock, response: AIMessage) -> ChatHuggingFace:
    with patch(
        "langchain_huggingface.chat_models.huggingface.ChatHuggingFace._resolve_model_id"
    ):
        return _StubChatHuggingFace(
            llm=mock_llm,
            tokenizer=MagicMock(),
            stub_response=response,
        )


def _tool_call_message(name: str, args: dict[str, Any]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": "call_1"}],
    )


def _bound_kwargs(runnable: Any) -> dict[str, Any]:
    assert isinstance(runnable, RunnableSequence)
    first = runnable.first
    assert isinstance(first, RunnableBinding)
    return first.kwargs


def _metadata_schema(runnable: Any) -> Any:
    return _bound_kwargs(runnable)["ls_structured_output_format"]["schema"]


def test_with_structured_output_function_calling_pydantic(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        _tool_call_message("_Joke", {"setup": "question", "punchline": "answer"}),
    )

    structured = chat.with_structured_output(_Joke)

    result = structured.invoke("Tell me a joke about cats.")
    assert isinstance(result, _Joke)
    assert result == _Joke(setup="question", punchline="answer")

    kwargs = _bound_kwargs(structured)
    assert kwargs["tools"][0]["function"]["name"] == "_Joke"
    assert kwargs["tool_choice"]["function"]["name"] == "_Joke"
    assert kwargs["ls_structured_output_format"]["kwargs"] == {
        "method": "function_calling"
    }
    assert _metadata_schema(structured) == convert_to_json_schema(_Joke)

    chunks = list(structured.stream("Tell me a joke about cats."))
    assert chunks
    assert all(isinstance(chunk, _Joke) for chunk in chunks)


def test_with_structured_output_function_calling_pydantic_v1(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        _tool_call_message("_JokeV1", {"setup": "question", "punchline": "answer"}),
    )

    structured = chat.with_structured_output(_JokeV1)

    result = structured.invoke("Tell me a joke about cats.")
    assert isinstance(result, _JokeV1)
    assert result.setup == "question"
    assert result.punchline == "answer"

    chunks = list(structured.stream("Tell me a joke about cats."))
    assert chunks
    assert all(isinstance(chunk, _JokeV1) for chunk in chunks)


def test_with_structured_output_optional_param(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        _tool_call_message("_OptionalJoke", {"setup": "question"}),
    )

    result = chat.with_structured_output(_OptionalJoke).invoke("Give me a setup.")

    assert isinstance(result, _OptionalJoke)
    assert result.setup == "question"
    assert result.punchline is None


def test_with_structured_output_pydantic_include_raw(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        _tool_call_message("_Joke", {"setup": "question", "punchline": "answer"}),
    )

    result = chat.with_structured_output(_Joke, include_raw=True).invoke("Tell me.")

    assert set(result) == {"raw", "parsed", "parsing_error"}
    assert isinstance(result["parsed"], _Joke)
    assert result["parsing_error"] is None


def test_with_structured_output_pydantic_include_raw_error(mock_llm: Mock) -> None:
    # Missing required `punchline` triggers a validation error.
    chat = _stub_chat(mock_llm, _tool_call_message("_Joke", {"setup": "question"}))

    result = chat.with_structured_output(_Joke, include_raw=True).invoke("Tell me.")

    assert result["parsed"] is None
    assert result["parsing_error"] is not None


def test_with_structured_output_json_schema_pydantic(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        AIMessage(content='{"setup": "question", "punchline": "answer"}'),
    )

    structured = chat.with_structured_output(_Joke, method="json_schema")

    result = structured.invoke("Tell me a joke about cats.")
    assert isinstance(result, _Joke)
    assert result == _Joke(setup="question", punchline="answer")

    kwargs = _bound_kwargs(structured)
    assert kwargs["response_format"] == {
        "type": "json_object",
        "schema": convert_to_json_schema(_Joke),
    }
    assert _metadata_schema(structured) == convert_to_json_schema(_Joke)

    chunks = list(structured.stream("Tell me a joke about cats."))
    assert chunks
    assert all(isinstance(chunk, _Joke) for chunk in chunks)


def test_with_structured_output_json_schema_pydantic_v1(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        AIMessage(content='{"setup": "question", "punchline": "answer"}'),
    )

    result = chat.with_structured_output(
        _JokeV1, method="json_schema"
    ).invoke("Tell me a joke about cats.")

    assert isinstance(result, _JokeV1)
    assert result.setup == "question"
    assert result.punchline == "answer"


def test_with_structured_output_json_mode_pydantic(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        AIMessage(content='{"setup": "question", "punchline": "answer"}'),
    )

    structured = chat.with_structured_output(_Joke, method="json_mode")

    result = structured.invoke("Tell me a joke about cats.")
    assert isinstance(result, _Joke)

    kwargs = _bound_kwargs(structured)
    assert kwargs["response_format"] == {"type": "json_object"}
    assert _metadata_schema(structured) == convert_to_json_schema(_Joke)

    chunks = list(structured.stream("Tell me a joke about cats."))
    assert chunks
    assert all(isinstance(chunk, _Joke) for chunk in chunks)


def test_with_structured_output_json_mode_pydantic_v1(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        AIMessage(content='{"setup": "question", "punchline": "answer"}'),
    )

    result = chat.with_structured_output(_JokeV1, method="json_mode").invoke(
        "Tell me a joke about cats."
    )

    assert isinstance(result, _JokeV1)
    assert result.setup == "question"


def test_with_structured_output_json_mode_without_schema(mock_llm: Mock) -> None:
    chat = _stub_chat(mock_llm, AIMessage(content='{"setup": "q", "punchline": "a"}'))

    structured = chat.with_structured_output(method="json_mode")

    result = structured.invoke("Return any JSON object.")
    assert result == {"setup": "q", "punchline": "a"}
    assert _metadata_schema(structured) is None


def test_with_structured_output_typeddict(mock_llm: Mock) -> None:
    chat = _stub_chat(
        mock_llm,
        _tool_call_message("_JokeDict", {"setup": "question", "punchline": "answer"}),
    )

    structured = chat.with_structured_output(_JokeDict)

    result = structured.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert result == {"setup": "question", "punchline": "answer"}
    assert _metadata_schema(structured) == convert_to_json_schema(_JokeDict)


def test_with_structured_output_json_schema_dict(mock_llm: Mock) -> None:
    schema = _Joke.model_json_schema()
    chat = _stub_chat(
        mock_llm,
        _tool_call_message("_Joke", {"setup": "question", "punchline": "answer"}),
    )

    structured = chat.with_structured_output(schema)

    result = structured.invoke("Tell me a joke about cats.")
    assert isinstance(result, dict)
    assert result == {"setup": "question", "punchline": "answer"}
    assert _metadata_schema(structured) == convert_to_json_schema(schema)
