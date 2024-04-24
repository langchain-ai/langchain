"""Test Alibaba Tongyi Chat Model."""
from typing import Any, cast

from langchain_core.callbacks import CallbackManager
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.pydantic_v1 import SecretStr
from pytest import CaptureFixture

from langchain_community.chat_models.tongyi import ChatTongyi
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

_FUNCTIONS: Any = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


def test_initialization() -> None:
    """Test chat model initialization."""
    for model in [
        ChatTongyi(model_name="qwen-turbo", api_key="xyz"),
        ChatTongyi(model="qwen-turbo", dashscope_api_key="xyz"),
    ]:
        assert model.model_name == "qwen-turbo"
        assert cast(SecretStr, model.dashscope_api_key).get_secret_value() == "xyz"


def test_api_key_is_string() -> None:
    llm = ChatTongyi(dashscope_api_key="secret-api-key")
    assert isinstance(llm.dashscope_api_key, SecretStr)


def test_api_key_masked_when_passed_via_constructor(
    capsys: CaptureFixture,
) -> None:
    llm = ChatTongyi(dashscope_api_key="secret-api-key")
    print(llm.dashscope_api_key, end="")  # noqa: T201
    captured = capsys.readouterr()

    assert captured.out == "**********"


def test_default_call() -> None:
    """Test default model call."""
    chat = ChatTongyi()
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_model() -> None:
    """Test model kwarg works."""
    chat = ChatTongyi(model="qwen-plus")
    response = chat.invoke([HumanMessage(content="Hello")])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_functions_call_thoughts() -> None:
    chat = ChatTongyi(model="qwen-plus")

    prompt_tmpl = "Use the given functions to answer following question: {input}"
    prompt_msgs = [
        HumanMessagePromptTemplate.from_template(prompt_tmpl),
    ]
    prompt = ChatPromptTemplate(messages=prompt_msgs)

    chain = prompt | chat.bind(functions=_FUNCTIONS)

    message = HumanMessage(content="What's the weather like in Shanghai today?")
    response = chain.batch([{"input": message}])
    assert isinstance(response[0], AIMessage)
    assert "tool_calls" in response[0].additional_kwargs


def test_multiple_history() -> None:
    """Tests multiple history works."""
    chat = ChatTongyi()

    response = chat.invoke(
        [
            HumanMessage(content="Hello."),
            AIMessage(content="Hello!"),
            HumanMessage(content="How are you doing?"),
        ]
    )
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_stream() -> None:
    """Test that stream works."""
    chat = ChatTongyi(streaming=True)
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


def test_multiple_messages() -> None:
    """Tests multiple messages works."""
    chat = ChatTongyi()
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
