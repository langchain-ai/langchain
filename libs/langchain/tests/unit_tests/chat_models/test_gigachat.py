import pytest
from gigachat.models import (
    ChatCompletion,
    ChatCompletionChunk,
    Choices,
    ChoicesChunk,
    Messages,
    MessagesChunk,
    Usage,
)

from langchain.chat_models.gigachat import (
    GigaChat,
    _convert_dict_to_message,
    _convert_message_to_dict,
)
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)

from ..callbacks.fake_callback_handler import (
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


def test__convert_dict_to_message_system():
    message = Messages(role="system", content="foo")
    expected = SystemMessage(content="foo")

    actual = _convert_dict_to_message(message)

    assert actual == expected


def test__convert_dict_to_message_human():
    message = Messages(role="user", content="foo")
    expected = HumanMessage(content="foo")

    actual = _convert_dict_to_message(message)

    assert actual == expected


def test__convert_dict_to_message_ai():
    message = Messages(role="assistant", content="foo")
    expected = AIMessage(content="foo")

    actual = _convert_dict_to_message(message)

    assert actual == expected


def test__convert_dict_to_message_type_error():
    message = Messages(role="user", content="foo")
    message.role = "bar"

    with pytest.raises(TypeError):
        _convert_dict_to_message(message)


def test__convert_message_to_dict_system():
    message = SystemMessage(content="foo")
    expected = Messages(role="system", content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


def test__convert_message_to_dict_human():
    message = HumanMessage(content="foo")
    expected = Messages(role="user", content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


def test__convert_message_to_dict_ai():
    message = AIMessage(content="foo")
    expected = Messages(role="assistant", content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


@pytest.mark.parametrize("role", ("system", "user", "assistant"))
def test__convert_message_to_dict_chat(role):
    message = ChatMessage(role=role, content="foo")
    expected = Messages(role=role, content="foo")

    actual = _convert_message_to_dict(message)

    assert actual == expected


def test__convert_message_to_dict_type_error():
    message = FunctionMessage(name="bar", content="foo")
    with pytest.raises(TypeError):
        _convert_message_to_dict(message)


@pytest.fixture
def chat_completion():
    return ChatCompletion(
        choices=[
            Choices(
                message=Messages(
                    role="assistant",
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
def chat_completion_stream():
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
def patch_gigachat(mocker, chat_completion, chat_completion_stream):
    mock = mocker.Mock()
    mock.chat.return_value = chat_completion
    mock.stream.return_value = chat_completion_stream

    mocker.patch("gigachat.GigaChat", return_value=mock)


@pytest.fixture
def patch_gigachat_achat(mocker, chat_completion):
    async def return_value_coroutine(value):
        return value

    mock = mocker.Mock()
    mock.achat.return_value = return_value_coroutine(chat_completion)

    mocker.patch("gigachat.GigaChat", return_value=mock)


@pytest.fixture
def patch_gigachat_astream(mocker, chat_completion_stream):
    async def return_value_async_generator(value):
        for chunk in value:
            yield chunk

    mock = mocker.Mock()
    mock.astream.return_value = return_value_async_generator(chat_completion_stream)

    mocker.patch("gigachat.GigaChat", return_value=mock)


def test_gigachat_predict(patch_gigachat):
    expected = "Bar Baz"

    llm = GigaChat()
    actual = llm.predict("bar")

    assert actual == expected


def test_gigachat_predict_stream(patch_gigachat):
    expected = "Bar Baz Stream"

    llm = GigaChat()
    callback_handler = FakeCallbackHandler()
    actual = llm.predict("bar", stream=True, callbacks=[callback_handler])

    assert actual == expected
    assert callback_handler.llm_streams == 2


@pytest.mark.asyncio()
async def test_gigachat_apredict(patch_gigachat_achat):
    expected = "Bar Baz"

    llm = GigaChat()
    actual = await llm.apredict("bar")

    assert actual == expected


@pytest.mark.asyncio()
async def test_gigachat_apredict_stream(patch_gigachat_astream):
    expected = "Bar Baz Stream"

    llm = GigaChat()
    callback_handler = FakeAsyncCallbackHandler()
    actual = await llm.apredict("bar", stream=True, callbacks=[callback_handler])

    assert actual == expected
    assert callback_handler.llm_streams == 2


def test_gigachat_stream(patch_gigachat):
    expected = [AIMessageChunk(content="Bar Baz"), AIMessageChunk(content=" Stream")]

    llm = GigaChat()
    actual = [chunk for chunk in llm.stream("bar")]

    assert actual == expected


@pytest.mark.asyncio()
async def test_gigachat_astream(patch_gigachat_astream):
    expected = [AIMessageChunk(content="Bar Baz"), AIMessageChunk(content=" Stream")]

    llm = GigaChat()
    actual = [chunk async for chunk in llm.astream("bar")]

    assert actual == expected
