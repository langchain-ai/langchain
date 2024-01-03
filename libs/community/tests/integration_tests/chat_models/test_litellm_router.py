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

answer1_text = "answer1"
chunk1_text = "chunk1"

model_list = [
    {
        "model_name": "azure-gpt-3.5-turbo",
        "litellm_params": {
            "model": "azure/fake-deployment-name-1",
            "api_key": "fakekeyvalue",
            "api_version": "",
            "api_base": "https://faketesturl/"
        },
    },
    {
        "model_name": "azure-gpt-3.5-turbo",
        "litellm_params": {
            "model": "azure/fake-deployment-name-2",
            "api_key": "fakekeyvalue",
            "api_version": "",
            "api_base": "https://faketesturl/"
        },
    }
]

def mock_completion_fn(**kwargs):
    from litellm import Usage
    base_result = {
            "choices": [
                    {
                        "index": 0,
                    }
            ],
            "created": 0,
            "id": "",
            "model": "gpt-3.5-turbo-0301",
            "object": "chat.completion",
        }
    if kwargs["stream"]:
        results = [deepcopy(base_result), deepcopy(base_result)]

        result_0_choice = results[0]["choices"][0]
        result_0_choice["delta"] = {
                "role": "assistent",
                "content": chunk1_text,
                "function_call": None,
            }
        result_0_choice["finish_reason"] = None
        # no usage here, since no usage from OpenAI API for streaming yet
        # https://community.openai.com/t/usage-info-in-api-responses/18862

        result_1_choice = results[1]["choices"][0]
        result_1_choice["delta"] = {}
        result_1_choice["finish_reason"] = "stop"
        # no usage here, since no usage from OpenAI API for streaming yet
        # https://community.openai.com/t/usage-info-in-api-responses/18862

        return results
    else:
        result = base_result
        choice = result["choices"][0]
        choice["message"] = {
            "content": answer1_text,
            "role": "assistant",
        }
        choice["finish_reason"] = "stop"
        result["usage"] = (Usage(completion_tokens=1, prompt_tokens=2, total_tokens=3))
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

async def mock_acompletion_fn(**kwargs):
    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(None, functools.partial(mock_completion_fn, **kwargs))
    if kwargs["stream"]:
        return AsyncIterator(results)
    else:
        return results

def setup_mocks() -> None:
    """Setup mocks."""
    import litellm
    litellm.completion = mock_completion_fn
    litellm.acompletion = mock_acompletion_fn

def get_test_router() -> None:
    """Get router for testing."""
    from litellm import Router
    return Router(
            model_list=model_list,
        )

@pytest.mark.scheduled
def test_litellm_router_call() -> None:
    """Test valid call to LiteLLM Router."""
    setup_mocks()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router})
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == answer1_text


@pytest.mark.scheduled
def test_litellm_router_generate() -> None:
    """Test generate method of LiteLLM Router."""
    from litellm import Usage
    setup_mocks()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router})
    chat_messages: List[List[BaseMessage]] = [
        [HumanMessage(content="How many toes do dogs have?")]
    ]
    messages_copy = [messages.copy() for messages in chat_messages]
    result: LLMResult = chat.generate(chat_messages)
    assert isinstance(result, LLMResult)
    for response in result.generations[0]:
        assert isinstance(response, ChatGeneration)
        assert isinstance(response.text, str)
        assert response.message.content == response.text
        assert response.message.content == answer1_text
    assert chat_messages == messages_copy
    assert result.llm_output["token_usage"] == Usage(completion_tokens=1, prompt_tokens=2, total_tokens=3)


@pytest.mark.scheduled
def test_litellm_router_streaming() -> None:
    """Test streaming tokens from LiteLLM Router."""
    setup_mocks()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router}, streaming=True)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == chunk1_text


@pytest.mark.scheduled
def test_litellm_router_streaming_callback() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    setup_mocks()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router},
        streaming=True,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")
    chat([message])
    assert callback_handler.llm_streams > 1

@pytest.mark.scheduled
async def test_async_litellm_router() -> None:
    """Test async generation."""
    from litellm import Usage
    setup_mocks()
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
            assert generation.message.content == answer1_text
    assert response.llm_output["token_usage"] == Usage(completion_tokens=1, prompt_tokens=2, total_tokens=3)


@pytest.mark.scheduled
async def test_async_litellm_router_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    setup_mocks()
    router = get_test_router()
    chat = ChatLiteLLMRouter(metadata={"router": router},
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
            assert generation.message.content == chunk1_text
