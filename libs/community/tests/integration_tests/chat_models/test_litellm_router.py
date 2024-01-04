"""Test LiteLLM Router API wrapper."""
import asyncio
import functools
from copy import deepcopy
from typing import List

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.litellm_router import ChatLiteLLMRouter
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

model_group = "gpt-4"
fake_model_prefix = "azure/fake-deployment-name-"
fake_api_key = "fakekeyvalue"
fake_api_version = "XXXX-XX-XX"
fake_api_base = "https://faketesturl/"
fake_chunks = ["This is ", "a fake answer."]
fake_answer = "".join(fake_chunks)
token_usage_key_name = "token_usage"

model_list = [
    {
        "model_name": model_group,
        "litellm_params": {
            "model": fake_model_prefix + "1",
            "api_key": fake_api_key,
            "api_version": fake_api_version,
            "api_base": fake_api_base,
        },
    },
    {
        "model_name": model_group,
        "litellm_params": {
            "model": fake_model_prefix + "2",
            "api_key": fake_api_key,
            "api_version": fake_api_version,
            "api_base": fake_api_base,
        },
    },
]


def fake_completion_fn(**kwargs):
    from litellm import Usage

    assert kwargs["model"].startswith(fake_model_prefix)
    assert kwargs["api_key"] == fake_api_key
    assert kwargs["api_version"] == fake_api_version
    assert kwargs["api_base"] == fake_api_base
    metadata = kwargs["metadata"]
    assert metadata["model_group"] == model_group
    assert metadata["deployment"].startswith(fake_model_prefix)

    base_result = {
        "choices": [
            {
                "index": 0,
            }
        ],
        "created": 0,
        "id": "",
        "model": model_group,
        "object": "chat.completion",
    }
    if kwargs["stream"]:
        results = []
        for chunk_index in range(0, len(fake_chunks)):
            result = deepcopy(base_result)
            choice = result["choices"][0]
            choice["delta"] = {
                "role": "assistent",
                "content": fake_chunks[chunk_index],
                "function_call": None,
            }
            choice["finish_reason"] = None
            # no usage here, since no usage from OpenAI API for streaming yet
            # https://community.openai.com/t/usage-info-in-api-responses/18862
            results.append(result)

        result = deepcopy(base_result)
        choice = result["choices"][0]
        choice["delta"] = {}
        choice["finish_reason"] = "stop"
        # no usage here, since no usage from OpenAI API for streaming yet
        # https://community.openai.com/t/usage-info-in-api-responses/18862
        results.append(result)

        return results
    else:
        result = base_result
        choice = result["choices"][0]
        choice["message"] = {
            "content": fake_answer,
            "role": "assistant",
        }
        choice["finish_reason"] = "stop"
        result["usage"] = Usage(completion_tokens=1, prompt_tokens=2, total_tokens=3)
        return result


class AsyncIterator:
    def __init__(self, seq):
        self.iter = iter(seq)

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self.iter)
        except StopIteration:
            raise StopAsyncIteration


async def fake_acompletion_fn(**kwargs):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None, functools.partial(fake_completion_fn, **kwargs)
    )
    if kwargs["stream"]:
        return AsyncIterator(results)
    else:
        return results


def setup_fakes() -> None:
    """Setup fakes."""
    import litellm

    # Turn off LiteLLM's built-in telemetry
    litellm.telemetry = False
    litellm.completion = fake_completion_fn
    litellm.acompletion = fake_acompletion_fn


def get_test_router() -> None:
    """Get router for testing."""
    from litellm import Router

    return Router(
        model_list=model_list,
    )


@pytest.mark.scheduled
def test_litellm_router_call() -> None:
    """Test valid call to LiteLLM Router."""
    setup_fakes()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router})
    message = HumanMessage(content="Hello")

    response = chat([message])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == fake_answer
    # no usage check here, since response is only an AIMessage


@pytest.mark.scheduled
def test_litellm_router_generate() -> None:
    """Test generate method of LiteLLM Router."""
    from litellm import Usage

    setup_fakes()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router})
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]

    result: LLMResult = chat.generate(chat_messages)

    assert isinstance(result, LLMResult)
    for generations in result.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.message.content == generation.text
            assert generation.message.content == fake_answer
    assert chat_messages == messages_copy
    assert result.llm_output[token_usage_key_name] == Usage(
        completion_tokens=1, prompt_tokens=2, total_tokens=3
    )


@pytest.mark.scheduled
def test_litellm_router_streaming() -> None:
    """Test streaming tokens from LiteLLM Router."""
    setup_fakes()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router}, streaming=True)
    message = HumanMessage(content="Hello")

    response = chat([message])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == fake_answer
    # no usage check here, since response is only an AIMessage


@pytest.mark.scheduled
def test_litellm_router_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    setup_fakes()
    router = get_test_router()
    chat = ChatLiteLLMRouter(
        metadata={"router": router},
        streaming=True,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")

    response = chat([message])

    assert callback_handler.llm_streams > 1
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == fake_answer
    # no usage check here, since response is only an AIMessage


@pytest.mark.scheduled
async def test_async_litellm_router() -> None:
    """Test async generation."""
    from litellm import Usage

    setup_fakes()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router})
    message = HumanMessage(content="Hello")

    response = await chat.agenerate([[message], [message]])

    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.message.content == generation.text
            assert generation.message.content == fake_answer
    assert response.llm_output[token_usage_key_name] == Usage(
        completion_tokens=2, prompt_tokens=4, total_tokens=6
    )


@pytest.mark.scheduled
async def test_async_litellm_router_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    setup_fakes()
    router = get_test_router()
    chat = ChatLiteLLMRouter(
        metadata={"router": router},
        streaming=True,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Hello")

    response = await chat.agenerate([[message], [message]])

    assert callback_handler.llm_streams > 0
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 1
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.message.content == generation.text
            assert generation.message.content == fake_answer
    # no usage check here, since no usage from OpenAI API for streaming yet
    # https://community.openai.com/t/usage-info-in-api-responses/18862
