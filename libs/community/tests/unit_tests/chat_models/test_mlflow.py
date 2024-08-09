import json
from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolCallChunk,
    ToolMessageChunk,
)
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import _PYDANTIC_MAJOR_VERSION, BaseModel
from langchain_core.tools import StructuredTool

from langchain_community.chat_models.mlflow import ChatMlflow


@pytest.fixture
def llm() -> ChatMlflow:
    return ChatMlflow(
        endpoint="databricks-meta-llama-3-70b-instruct", target_uri="databricks"
    )


@pytest.fixture
def model_input() -> List[BaseMessage]:
    data = [
        {
            "role": "system",
            "content": "You are a helpful assistant.",
        },
        {"role": "user", "content": "36939 * 8922.4"},
    ]
    return [ChatMlflow._convert_dict_to_message(value) for value in data]


@pytest.fixture
def mock_prediction() -> dict:
    return {
        "id": "chatcmpl_id",
        "object": "chat.completion",
        "created": 1721875529,
        "model": "meta-llama-3.1-70b-instruct-072424",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "To calculate the result of 36939 multiplied by 8922.4, "
                    "I get:\n\n36939 x 8922.4 = 329,511,111.6",
                },
                "finish_reason": "stop",
                "logprobs": None,
            }
        ],
        "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
    }


@pytest.fixture
def mock_predict_stream_result() -> List[dict]:
    return [
        {
            "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
            "object": "chat.completion.chunk",
            "created": 1721877054,
            "model": "meta-llama-3.1-70b-instruct-072424",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "36939"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 20, "total_tokens": 50},
        },
        {
            "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
            "object": "chat.completion.chunk",
            "created": 1721877054,
            "model": "meta-llama-3.1-70b-instruct-072424",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "x"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 22, "total_tokens": 52},
        },
        {
            "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
            "object": "chat.completion.chunk",
            "created": 1721877054,
            "model": "meta-llama-3.1-70b-instruct-072424",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "8922.4"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 24, "total_tokens": 54},
        },
        {
            "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
            "object": "chat.completion.chunk",
            "created": 1721877054,
            "model": "meta-llama-3.1-70b-instruct-072424",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": " = "},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 28, "total_tokens": 58},
        },
        {
            "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
            "object": "chat.completion.chunk",
            "created": 1721877054,
            "model": "meta-llama-3.1-70b-instruct-072424",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "329,511,111.6"},
                    "finish_reason": None,
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 30, "total_tokens": 60},
        },
        {
            "id": "chatcmpl_bb1fce87-f14e-4ae1-ac22-89facc74898a",
            "object": "chat.completion.chunk",
            "created": 1721877054,
            "model": "meta-llama-3.1-70b-instruct-072424",
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": "stop",
                    "logprobs": None,
                }
            ],
            "usage": {"prompt_tokens": 30, "completion_tokens": 36, "total_tokens": 66},
        },
    ]


@pytest.mark.requires("mlflow")
def test_chat_mlflow_predict(
    llm: ChatMlflow, model_input: List[BaseMessage], mock_prediction: dict
) -> None:
    mock_client = MagicMock()
    llm._client = mock_client

    def mock_predict(*args: Any, **kwargs: Any) -> Any:
        return mock_prediction

    mock_client.predict = mock_predict
    res = llm.invoke(model_input)
    assert res.content == mock_prediction["choices"][0]["message"]["content"]


@pytest.mark.requires("mlflow")
def test_chat_mlflow_stream(
    llm: ChatMlflow,
    model_input: List[BaseMessage],
    mock_predict_stream_result: List[dict],
) -> None:
    mock_client = MagicMock()
    llm._client = mock_client

    def mock_stream(*args: Any, **kwargs: Any) -> Any:
        yield from mock_predict_stream_result

    mock_client.predict_stream = mock_stream
    for i, res in enumerate(llm.stream(model_input)):
        assert (
            res.content
            == mock_predict_stream_result[i]["choices"][0]["delta"]["content"]
        )


@pytest.mark.requires("mlflow")
@pytest.mark.skipif(
    _PYDANTIC_MAJOR_VERSION < 2,
    reason="The tool mock is not compatible with pydantic 1.x",
)
def test_chat_mlflow_bind_tools(
    llm: ChatMlflow, mock_predict_stream_result: List[dict]
) -> None:
    mock_client = MagicMock()
    llm._client = mock_client

    def mock_stream(*args: Any, **kwargs: Any) -> Any:
        yield from mock_predict_stream_result

    mock_client.predict_stream = mock_stream

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Make sure to use tool for information.",
            ),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    def mock_func(*args: Any, **kwargs: Any) -> str:
        return "36939 x 8922.4 = 329,511,111.6"

    tools = [
        StructuredTool(
            name="name",
            description="description",
            args_schema=BaseModel,
            func=mock_func,
        )
    ]
    agent = create_tool_calling_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # type: ignore[arg-type]
    result = agent_executor.invoke({"input": "36939 * 8922.4"})
    assert result["output"] == "36939x8922.4 = 329,511,111.6"


def test_convert_dict_to_message_human() -> None:
    message = {"role": "user", "content": "foo"}
    result = ChatMlflow._convert_dict_to_message(message)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test_convert_dict_to_message_ai() -> None:
    message = {"role": "assistant", "content": "foo"}
    result = ChatMlflow._convert_dict_to_message(message)
    expected_output = AIMessage(content="foo")
    assert result == expected_output

    tool_calls = [
        {
            "id": "call_fb5f5e1a-bac0-4422-95e9-d06e6022ad12",
            "type": "function",
            "function": {
                "name": "main__test__python_exec",
                "arguments": '{"code": "result = 36939 * 8922.4" }',
            },
        }
    ]
    message_with_tools: Dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": tool_calls,
    }
    result = ChatMlflow._convert_dict_to_message(message_with_tools)
    expected_output = AIMessage(
        content="",
        additional_kwargs={"tool_calls": tool_calls},
        id="call_fb5f5e1a-bac0-4422-95e9-d06e6022ad12",
        tool_calls=[
            {
                "name": tool_calls[0]["function"]["name"],  # type: ignore[index]
                "args": json.loads(tool_calls[0]["function"]["arguments"]),  # type: ignore[index]
                "id": "call_fb5f5e1a-bac0-4422-95e9-d06e6022ad12",
                "type": "tool_call",
            }
        ],
    )


def test_convert_dict_to_message_system() -> None:
    message = {"role": "system", "content": "foo"}
    result = ChatMlflow._convert_dict_to_message(message)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output


def test_convert_dict_to_message_chat() -> None:
    message = {"role": "any_role", "content": "foo"}
    result = ChatMlflow._convert_dict_to_message(message)
    expected_output = ChatMessage(content="foo", role="any_role")
    assert result == expected_output


def test_convert_delta_to_message_chunk_ai() -> None:
    delta = {"role": "assistant", "content": "foo"}
    result = ChatMlflow._convert_delta_to_message_chunk(delta, "default_role")
    expected_output = AIMessageChunk(content="foo")
    assert result == expected_output

    delta_with_tools: Dict[str, Any] = {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"index": 0, "function": {"arguments": " }"}}],
    }
    result = ChatMlflow._convert_delta_to_message_chunk(delta_with_tools, "role")
    expected_output = AIMessageChunk(
        content="",
        additional_kwargs={"tool_calls": delta_with_tools["tool_calls"]},
        id=None,
        tool_call_chunks=[ToolCallChunk(name=None, args=" }", id=None, index=0)],
    )
    assert result == expected_output


def test_convert_delta_to_message_chunk_tool() -> None:
    delta = {
        "role": "tool",
        "content": "foo",
        "tool_call_id": "tool_call_id",
        "id": "some_id",
    }
    result = ChatMlflow._convert_delta_to_message_chunk(delta, "default_role")
    expected_output = ToolMessageChunk(
        content="foo", id="some_id", tool_call_id="tool_call_id"
    )
    assert result == expected_output


def test_convert_delta_to_message_chunk_human() -> None:
    delta = {
        "role": "user",
        "content": "foo",
    }
    result = ChatMlflow._convert_delta_to_message_chunk(delta, "default_role")
    expected_output = HumanMessageChunk(content="foo")
    assert result == expected_output


def test_convert_delta_to_message_chunk_system() -> None:
    delta = {
        "role": "system",
        "content": "foo",
    }
    result = ChatMlflow._convert_delta_to_message_chunk(delta, "default_role")
    expected_output = SystemMessageChunk(content="foo")
    assert result == expected_output


def test_convert_delta_to_message_chunk_chat() -> None:
    delta = {
        "role": "any_role",
        "content": "foo",
    }
    result = ChatMlflow._convert_delta_to_message_chunk(delta, "default_role")
    expected_output = ChatMessageChunk(content="foo", role="any_role")
    assert result == expected_output


def test_convert_message_to_dict_human() -> None:
    human_message = HumanMessage(content="foo")
    result = ChatMlflow._convert_message_to_dict(human_message)
    expected_output = {"role": "user", "content": "foo"}
    assert result == expected_output


def test_convert_message_to_dict_system() -> None:
    system_message = SystemMessage(content="foo")
    result = ChatMlflow._convert_message_to_dict(system_message)
    expected_output = {"role": "system", "content": "foo"}
    assert result == expected_output


def test_convert_message_to_dict_ai() -> None:
    ai_message = AIMessage(content="foo")
    result = ChatMlflow._convert_message_to_dict(ai_message)
    expected_output = {"role": "assistant", "content": "foo"}
    assert result == expected_output

    ai_message = AIMessage(
        content="",
        tool_calls=[{"name": "name", "args": {}, "id": "id", "type": "tool_call"}],
    )
    result = ChatMlflow._convert_message_to_dict(ai_message)
    expected_output_with_tools: Dict[str, Any] = {
        "content": None,
        "role": "assistant",
        "tool_calls": [
            {
                "type": "function",
                "id": "id",
                "function": {"name": "name", "arguments": "{}"},
            }
        ],
    }
    assert result == expected_output_with_tools


def test_convert_message_to_dict_tool() -> None:
    tool_message = ToolMessageChunk(
        content="foo", id="some_id", tool_call_id="tool_call_id"
    )
    result = ChatMlflow._convert_message_to_dict(tool_message)
    expected_output = {
        "role": "tool",
        "content": "foo",
        "tool_call_id": "tool_call_id",
    }
    assert result == expected_output


def test_convert_message_to_dict_function() -> None:
    with pytest.raises(ValueError):
        ChatMlflow._convert_message_to_dict(FunctionMessage(content="", name="name"))
