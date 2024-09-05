# flake8: noqa: I001
from typing import Any, AsyncGenerator, Iterable, List

import pytest
from pytest_mock import MockerFixture

from gigachat.models import (
    ChatCompletion,
    ChatCompletionChunk,
    Choices,
    ChoicesChunk,
    MessagesRole,
    Messages,
    MessagesChunk,
    Usage,
)
from langchain_community.llms.gigachat import GigaChat

from ..callbacks.fake_callback_handler import (
    FakeAsyncCallbackHandler,
    FakeCallbackHandler,
)


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
    expected = ["Bar Baz", " Stream"]

    llm = GigaChat()
    actual = [chunk for chunk in llm.stream("bar")]

    assert actual == expected


@pytest.mark.asyncio()
async def test_gigachat_astream(patch_gigachat_astream: None) -> None:
    expected = ["Bar Baz", " Stream"]

    llm = GigaChat()
    actual = [chunk async for chunk in llm.astream("bar")]

    assert actual == expected
