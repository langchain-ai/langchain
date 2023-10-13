"""Test Baidu Qianfan Chat Endpoint."""

from typing import Any

from langchain.callbacks.manager import CallbackManager
from langchain.chains.openai_functions import (
    create_openai_fn_chain,
)
from langchain.chat_models.baidu_qianfan_endpoint import QianfanChatEndpoint
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    FunctionMessage,
    HumanMessage,
    LLMResult,
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


def test_default_call() -> None:
    """Test default model(`ERNIE-Bot`) call."""
    chat = QianfanChatEndpoint()
    response = chat(messages=[HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model() -> None:
    """Test model kwarg works."""
    chat = QianfanChatEndpoint(model="BLOOMZ-7B")
    response = chat(messages=[HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model_param() -> None:
    """Test model params works."""
    chat = QianfanChatEndpoint()
    response = chat(model="BLOOMZ-7B", messages=[HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_endpoint() -> None:
    """Test user custom model deployments like some open source models."""
    chat = QianfanChatEndpoint(endpoint="qianfan_bloomz_7b_compressed")
    response = chat(messages=[HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_endpoint_param() -> None:
    """Test user custom model deployments like some open source models."""
    chat = QianfanChatEndpoint()
    response = chat(
        messages=[
            HumanMessage(endpoint="qianfan_bloomz_7b_compressed", content="Hello")
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = QianfanChatEndpoint()

    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_stream() -> None:
    """Test that stream works."""
    chat = QianfanChatEndpoint(streaming=True)
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    response = chat(
        messages=[
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="Who are you?"),
        ],
        stream=True,
        callbacks=callback_manager,
    )
    assert callback_handler.llm_streams > 0
    assert isinstance(response.content, str)


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = QianfanChatEndpoint()
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
    chat = QianfanChatEndpoint(model="ERNIE-Bot")

    prompt_tmpl = "Use the given functions to answer following question: {input}"
    prompt_msgs = [
        HumanMessagePromptTemplate.from_template(prompt_tmpl),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)

    chain = create_openai_fn_chain(
        _FUNCTIONS,
        chat,
        prompt,
        output_parser=None,
    )

    message = HumanMessage(content="What's the temperature in Shanghai today?")
    response = chain.generate([{"input": message}])
    assert isinstance(response.generations[0][0], ChatGeneration)
    assert isinstance(response.generations[0][0].message, AIMessage)
    assert "function_call" in response.generations[0][0].message.additional_kwargs


def test_functions_call() -> None:
    chat = QianfanChatEndpoint(model="ERNIE-Bot")

    prompt = ChatPromptTemplate(
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
    llm_chain = create_openai_fn_chain(
        _FUNCTIONS,
        chat,
        prompt,
        output_parser=None,
    )
    resp = llm_chain.generate([{}])
    assert isinstance(resp, LLMResult)
