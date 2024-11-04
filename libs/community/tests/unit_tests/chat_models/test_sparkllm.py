import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers.openai_tools import (
    parse_tool_call,
)

from langchain_community.chat_models.sparkllm import (
    ChatSparkLLM,
    convert_dict_to_message,
    convert_message_to_dict,
)


def test__convert_dict_to_message_human() -> None:
    message_dict = {"role": "user", "content": "foo"}
    result = convert_dict_to_message(message_dict)
    expected_output = HumanMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_ai() -> None:
    message_dict = {"role": "assistant", "content": "foo"}
    result = convert_dict_to_message(message_dict)
    expected_output = AIMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_other_role() -> None:
    message_dict = {"role": "system", "content": "foo"}
    result = convert_dict_to_message(message_dict)
    expected_output = SystemMessage(content="foo")
    assert result == expected_output


def test__convert_dict_to_message_function_call() -> None:
    raw_function_calls = [
        {
            "function": {
                "name": "get_current_weather",
                "arguments": '{"location": "Boston", "unit": "fahrenheit"}',
            },
            "type": "function",
        }
    ]
    message_dict = {
        "role": "assistant",
        "content": "foo",
        "tool_calls": raw_function_calls,
    }
    result = convert_dict_to_message(message_dict)

    tool_calls = [
        parse_tool_call(raw_tool_call, return_id=True)
        for raw_tool_call in raw_function_calls
    ]
    expected_output = AIMessage(
        content="foo",
        additional_kwargs={"tool_calls": raw_function_calls},
        tool_calls=tool_calls,
        invalid_tool_calls=[],
    )
    assert result == expected_output


def test__convert_message_to_dict_human() -> None:
    message = HumanMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"role": "user", "content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_ai() -> None:
    message = AIMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"role": "assistant", "content": "foo"}
    assert result == expected_output


def test__convert_message_to_dict_system() -> None:
    message = SystemMessage(content="foo")
    result = convert_message_to_dict(message)
    expected_output = {"role": "system", "content": "foo"}
    assert result == expected_output


@pytest.mark.requires("websocket")
def test__chat_spark_llm_initialization() -> None:
    chat = ChatSparkLLM(
        app_id="IFLYTEK_SPARK_APP_ID",
        api_key="IFLYTEK_SPARK_API_KEY",
        api_secret="IFLYTEK_SPARK_API_SECRET",
        api_url="IFLYTEK_SPARK_API_URL",
        model="IFLYTEK_SPARK_LLM_DOMAIN",
        timeout=40,
        temperature=0.1,
        top_k=3,
    )
    assert chat.spark_app_id == "IFLYTEK_SPARK_APP_ID"
    assert chat.spark_api_key == "IFLYTEK_SPARK_API_KEY"
    assert chat.spark_api_secret == "IFLYTEK_SPARK_API_SECRET"
    assert chat.spark_api_url == "IFLYTEK_SPARK_API_URL"
    assert chat.spark_llm_domain == "IFLYTEK_SPARK_LLM_DOMAIN"
    assert chat.request_timeout == 40
    assert chat.temperature == 0.1
    assert chat.top_k == 3
