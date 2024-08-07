"""Test Baidu Qianfan Chat Endpoint."""

from typing import Any, cast

import pytest
from langchain_core.callbacks import CallbackManager
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    BaseMessageChunk,
    FunctionMessage,
    HumanMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture, MonkeyPatch

from langchain_community.chat_models.baidu_qianfan_endpoint import (
    QianfanChatEndpoint,
)
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

_FUNCTIONS: Any = [
    {
        "name": "format_person_info",
        "description": (
            "Output formatter. Should always be used to format your response to the"
            " user."
        ),
        "parameters": {
            "title": "Person",
            "description": "Identifying information about a person.",
            "type": "object",
            "properties": {
                "name": {
                    "title": "Name",
                    "description": "The person's name",
                    "type": "string",
                },
                "age": {
                    "title": "Age",
                    "description": "The person's age",
                    "type": "integer",
                },
                "fav_food": {
                    "title": "Fav Food",
                    "description": "The person's favorite food",
                    "type": "string",
                },
            },
            "required": ["name", "age"],
        },
    },
    {
        "name": "get_current_temperature",
        "description": ("Used to get the location's temperature."),
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "city name",
                },
                "unit": {
                    "type": "string",
                    "enum": ["centigrade", "Fahrenheit"],
                },
            },
            "required": ["location", "unit"],
        },
        "responses": {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "integer",
                    "description": "city temperature",
                },
                "unit": {
                    "type": "string",
                    "enum": ["centigrade", "Fahrenheit"],
                },
            },
        },
    },
]


def test_initialization() -> None:
    """Test chat model initialization."""

    for model in [
        QianfanChatEndpoint(model="BLOOMZ-7B", timeout=40),  # type: ignore[call-arg]
        QianfanChatEndpoint(model="BLOOMZ-7B", request_timeout=40),  # type: ignore[call-arg]
    ]:
        assert model.model == "BLOOMZ-7B"
        assert model.request_timeout == 40


def test_default_call() -> None:
    """Test default model.invoke(`ERNIE-Bot`) call."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model() -> None:
    """Test model kwarg works."""
    chat = QianfanChatEndpoint(model="BLOOMZ-7B")  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model_param() -> None:
    """Test model params works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")], model="BLOOMZ-7B")
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_endpoint() -> None:
    """Test user custom model deployments like some open source models."""
    chat = QianfanChatEndpoint(endpoint="qianfan_bloomz_7b_compressed")  # type: ignore[call-arg]
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_endpoint_param() -> None:
    """Test user custom model deployments like some open source models."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.invoke(
        [HumanMessage(endpoint="qianfan_bloomz_7b_compressed", content="Hello")]  # type: ignore[call-arg]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_generate() -> None:
    """Tests chat generate works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = chat.generate(
        [
            [
                HumanMessage(content="Hello."),
                AIMessage(content="Hello!"),
                HumanMessage(content="How are you doing?"),
            ]
        ]
    )
    assert isinstance(response, LLMResult)
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)


def test_stream() -> None:
    """Test that stream works."""
    chat = QianfanChatEndpoint(streaming=True)  # type: ignore[call-arg]
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ],
        stream=True,
        config={"callbacks": callback_manager},
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)

    res = chat.stream(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ]
    )

    assert len(list(res)) >= 1


@pytest.mark.asyncio
async def test_async_invoke() -> None:
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    res = await chat.ainvoke([HumanMessage(content="Hello")])
    assert isinstance(res, BaseMessage)
    assert res.content != ""


@pytest.mark.asyncio
async def test_async_generate() -> None:
    """Tests chat agenerate works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    response = await chat.agenerate(
        [
            [
                HumanMessage(content="Hello."),
                AIMessage(content="Hello!"),
                HumanMessage(content="How are you doing?"),
            ]
        ]
    )
    assert isinstance(response, LLMResult)
    for generations in response.generations:
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)


@pytest.mark.asyncio
async def test_async_stream() -> None:
    chat = QianfanChatEndpoint(streaming=True)  # type: ignore[call-arg]
    async for token in chat.astream(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ]
    ):
        assert isinstance(token, BaseMessageChunk)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    message = HumanMessage(content="Hi, how are you.")
    response = chat.generate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


def test_functions_call_thoughts() -> None:
    chat = QianfanChatEndpoint(model="ERNIE-Bot")  # type: ignore[call-arg]

    prompt_tmpl = "Use the given functions to answer following question: {input}"
    prompt_msgs = [
        HumanMessagePromptTemplate.from_template(prompt_tmpl),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)  # type: ignore[arg-type, call-arg]

    chain = prompt | chat.bind(functions=_FUNCTIONS)

    message = HumanMessage(content="What's the temperature in Shanghai today?")
    response = chain.batch([{"input": message}])
    assert isinstance(response[0], AIMessage)
    assert "function_call" in response[0].additional_kwargs


def test_functions_call() -> None:
    chat = QianfanChatEndpoint(model="ERNIE-Bot")  # type: ignore[call-arg]

    prompt = ChatPromptTemplate(  # type: ignore[call-arg]
        messages=[
            HumanMessage(content="What's the temperature in Shanghai today?"),
            AIMessage(
                content="",
                additional_kwargs={
                    "function_call": {
                        "name": "get_current_temperature",
                        "thoughts": "i will use get_current_temperature "
                        "to resolve the questions",
                        "arguments": '{"location":"Shanghai","unit":"centigrade"}',
                    }
                },
            ),
            FunctionMessage(
                name="get_current_weather",
                content='{"temperature": "25", \
                                "unit": "摄氏度", "description": "晴朗"}',
            ),
        ]
    )
    chain = prompt | chat.bind(functions=_FUNCTIONS)
    resp = chain.invoke({})
    assert isinstance(resp, AIMessage)


def test_rate_limit() -> None:
    chat = QianfanChatEndpoint(model="ERNIE-Bot", init_kwargs={"query_per_second": 2})  # type: ignore[call-arg]
    assert (
        chat.client._client._rate_limiter._internal_qps_rate_limiter._sync_limiter._query_per_second
        == 1.8
    )
    responses = chat.batch(
        [
            [HumanMessage(content="Hello")],
            [HumanMessage(content="who are you")],
            [HumanMessage(content="what is baidu")],
        ]
    )
    for res in responses:
        assert isinstance(res, BaseMessage)
        assert isinstance(res.content, str)


def test_qianfan_key_masked_when_passed_from_env(
    monkeypatch: MonkeyPatch, capsys: CaptureFixture
) -> None:
    """Test initialization with an API key provided via an env variable"""
    monkeypatch.setenv("QIANFAN_AK", "test-api-key")
    monkeypatch.setenv("QIANFAN_SK", "test-secret-key")

    chat = QianfanChatEndpoint()  # type: ignore[call-arg]
    print(chat.qianfan_ak, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.qianfan_sk, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"


def test_qianfan_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    """Test initialization with an API key provided via the initializer"""
    chat = QianfanChatEndpoint(  # type: ignore[call-arg]
        qianfan_ak="test-api-key",  # type: ignore[arg-type]
        qianfan_sk="test-secret-key",  # type: ignore[arg-type]
    )
    print(chat.qianfan_ak, end="")  # noqa: T201
    captured = capsys.readouterr()
    assert captured.out == "**********"

    print(chat.qianfan_sk, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_uses_actual_secret_value_from_secret_str() -> None:
    """Test that actual secret is retrieved using `.get_secret_value()`."""
    chat = QianfanChatEndpoint(  # type: ignore[call-arg]
        qianfan_ak="test-api-key",  # type: ignore[arg-type]
        qianfan_sk="test-secret-key",  # type: ignore[arg-type]
    )
    assert cast(SecretStr, chat.qianfan_ak).get_secret_value() == "test-api-key"
    assert cast(SecretStr, chat.qianfan_sk).get_secret_value() == "test-secret-key"


def test_init_api_key_param() -> None:
    """Test the standardized parameters -- api_key and secret_key"""
    for chat in [
        QianfanChatEndpoint(  # type: ignore[call-arg]
            api_key="test-api-key",  # type: ignore[arg-type]
            secret_key="test-secret-key",  # type: ignore[arg-type]
        ),
        QianfanChatEndpoint(  # type: ignore[call-arg]
            qianfan_ak="test-api-key",  # type: ignore[arg-type]
            qianfan_sk="test-secret-key",  # type: ignore[arg-type]
        ),
    ]:
        assert cast(SecretStr, chat.qianfan_ak).get_secret_value() == "test-api-key"
        assert cast(SecretStr, chat.qianfan_sk).get_secret_value() == "test-secret-key"
