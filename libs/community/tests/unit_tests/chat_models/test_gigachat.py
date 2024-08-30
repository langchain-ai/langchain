# flake8: noqa: I001
from typing import Any, AsyncGenerator, Iterable, List

import pytest
from gigachat.models import (
    ChatCompletion,
    ChatCompletionChunk,
    Choices,
    ChoicesChunk,
    Messages,
    MessagesChunk,
    MessagesRole,
    Usage,
)
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_community.chat_models.gigachat import (
    GigaChat,
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from pytest_mock import MockerFixture
from tests.unit_tests.stubs import AnyStr

from ..callbacks.fake_callback_handler import (
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


def test__convert_dict_to_message_system() -> None:
    message = Messages(id=None, role=MessagesRole.SYSTEM, content="foo")
    expected = SystemMessage(content="foo")

    actual = _convert_dict_to_message(message)

    assert actual == expected


def test__convert_dict_to_message_human() -> None:
    message = Messages(id=None, role=MessagesRole.USER, content="foo")
    expected = HumanMessage(content="foo")

    actual = _convert_dict_to_message(message)

    assert actual == expected


def test__convert_dict_to_message_ai() -> None:
    message = Messages(id=None, role=MessagesRole.ASSISTANT, content="foo")
    expected = AIMessage(content="foo")

    actual = _convert_dict_to_message(message)

    assert actual == expected


def test__convert_message_to_dict_system() -> None:
    message = SystemMessage(content="foo")
    expected = Messages(id=None, role=MessagesRole.SYSTEM, content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


def test__convert_message_to_dict_human() -> None:
    message = HumanMessage(content="foo")
    expected = Messages(id=None, role=MessagesRole.USER, content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


def test__convert_message_to_dict_ai() -> None:
    message = AIMessage(content="foo")
    expected = Messages(id=None, role=MessagesRole.ASSISTANT, content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


@pytest.mark.parametrize(
    "pairs", (("{}", "{}"), ("abc", '"abc"'), (123, "123"), ("[]", "[]"))
)
def test__convert_message_to_dict_function(pairs: Any) -> None:
    """Checks if string, that was not JSON was converted to JSON"""
    message = FunctionMessage(content=pairs[0], id="1", name="func")
    expected = Messages(id=None, role=MessagesRole.FUNCTION, content=pairs[1])

    actual = _convert_message_to_dict(message)

    assert actual == expected


@pytest.mark.parametrize(
    "role",
    (
        MessagesRole.SYSTEM,
        MessagesRole.USER,
        MessagesRole.ASSISTANT,
        MessagesRole.FUNCTION,
    ),
)
def test__convert_message_to_dict_chat(role: MessagesRole) -> None:
    message = ChatMessage(role=role, content="foo")
    expected = Messages(id=None, role=role, content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


@pytest.fixture
def chat_completion() -> ChatCompletion:
    return ChatCompletion(
        choices=[
            Choices(
                message=Messages(
                    id=None,
                    role=MessagesRole.ASSISTANT,
                    content="Bar Baz",
                ),
                index=0,
                finish_reason="stop",
            ),
        ],
        created=1678878333,
        model="GigaChat:v1.2.19.2",
        usage=Usage(
            prompt_tokens=18,
            completion_tokens=68,
            total_tokens=86,
        ),
        object="chat.completion",
    )


@pytest.fixture
def chat_completion_stream() -> List[ChatCompletionChunk]:
    return [
        ChatCompletionChunk(
            choices=[
                ChoicesChunk(
                    delta=MessagesChunk(content="Bar Baz"),
                    index=0,
                ),
            ],
            created=1695802242,
            model="GigaChat:v1.2.19.2",
            object="chat.completion",
        ),
        ChatCompletionChunk(
            choices=[
                ChoicesChunk(
                    delta=MessagesChunk(content=" Stream"),
                    index=0,
                    finish_reason="stop",
                ),
            ],
            created=1695802242,
            model="GigaChat:v1.2.19.2",
            object="chat.completion",
        ),
    ]


@pytest.fixture
def patch_gigachat(
    mocker: MockerFixture,
    chat_completion: ChatCompletion,
    chat_completion_stream: List[ChatCompletionChunk],
) -> None:
    mock = mocker.Mock()
    mock.chat.return_value = chat_completion
    mock.stream.return_value = chat_completion_stream

    mocker.patch("gigachat.GigaChat", return_value=mock)


@pytest.fixture
def patch_gigachat_achat(
    mocker: MockerFixture, chat_completion: ChatCompletion
) -> None:
    async def return_value_coroutine(value: Any) -> Any:
        return value

    mock = mocker.Mock()
    mock.achat.return_value = return_value_coroutine(chat_completion)

    mocker.patch("gigachat.GigaChat", return_value=mock)


@pytest.fixture
def patch_gigachat_astream(
    mocker: MockerFixture, chat_completion_stream: List[ChatCompletionChunk]
) -> None:
    async def return_value_async_generator(value: Iterable) -> AsyncGenerator:
        for chunk in value:
            yield chunk

    mock = mocker.Mock()
    mock.astream.return_value = return_value_async_generator(chat_completion_stream)

    mocker.patch("gigachat.GigaChat", return_value=mock)


def test_gigachat_predict(patch_gigachat: None) -> None:
    expected = "Bar Baz"

    llm = GigaChat()
    actual = llm.predict("bar")

    assert actual == expected


def test_gigachat_predict_stream(patch_gigachat: None) -> None:
    expected = "Bar Baz Stream"

    llm = GigaChat()
    callback_handler = FakeCallbackHandler()
    actual = llm.predict("bar", stream=True, callbacks=[callback_handler])

    assert actual == expected
    assert callback_handler.llm_streams == 2


@pytest.mark.asyncio()
async def test_gigachat_apredict(patch_gigachat_achat: None) -> None:
    expected = "Bar Baz"

    llm = GigaChat()
    actual = await llm.apredict("bar")

    assert actual == expected


@pytest.mark.asyncio()
async def test_gigachat_apredict_stream(patch_gigachat_astream: None) -> None:
    expected = "Bar Baz Stream"

    llm = GigaChat()
    callback_handler = FakeAsyncCallbackHandler()
    actual = await llm.apredict("bar", stream=True, callbacks=[callback_handler])

    assert actual == expected
    assert callback_handler.llm_streams == 2


def test_gigachat_stream(patch_gigachat: None) -> None:
    expected = [
        AIMessageChunk(content="Bar Baz", id=AnyStr()),
        AIMessageChunk(
            content=" Stream", response_metadata={"finish_reason": "stop"}, id=AnyStr()
        ),
    ]

    llm = GigaChat()
    actual = [chunk for chunk in llm.stream("bar")]

    assert actual == expected


@pytest.mark.asyncio()
async def test_gigachat_astream(patch_gigachat_astream: None) -> None:
    expected = [
        AIMessageChunk(content="Bar Baz", id=AnyStr()),
        AIMessageChunk(
            content=" Stream", response_metadata={"finish_reason": "stop"}, id=AnyStr()
        ),
    ]

    llm = GigaChat()
    actual = [chunk async for chunk in llm.astream("bar")]

    assert actual == expected


def test_gigachat_build_payload_existing_parameter() -> None:
    llm = GigaChat()
    payload = llm._build_payload([], max_tokens=1)
    assert payload.max_tokens == 1


def test_gigachat_build_payload_non_existing_parameter() -> None:
    llm = GigaChat()
    payload = llm._build_payload([], fake_parameter=1)
    assert getattr(payload, "fake_param", None) is None


async def test_gigachat_bind_without_description() -> None:
    class Person(BaseModel):
        name: str = Field(..., title="Name", description="The person's name")

    llm = GigaChat()
    with pytest.raises(ValueError):
        llm.bind_functions(functions=[Person], function_call="Person")
    with pytest.raises(ValueError):
        llm.bind_tools(tools=[Person], tool_choice="Person")


async def test_gigachat_bind_with_description() -> None:
    class Person(BaseModel):
        """Simple description"""

        name: str = Field(..., title="Name")

    llm = GigaChat()
    llm.bind_functions(functions=[Person], function_call="Person")
    llm.bind_tools(tools=[Person], tool_choice="Person")
