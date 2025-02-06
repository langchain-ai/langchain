"""Test LiteLLM Router API wrapper."""

import asyncio
import queue
import threading
from copy import deepcopy
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Dict,
    Generator,
    List,
    Tuple,
    Union,
    cast,
)

import pytest
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_community.chat_models.litellm_router import ChatLiteLLMRouter
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

model_group_gpt4 = "gpt-4"
model_group_to_test = "gpt-35-turbo"
fake_model_prefix = "azure/fake-deployment-name-"
fake_models_names = [fake_model_prefix + suffix for suffix in ["1", "2"]]
fake_api_key = "fakekeyvalue"
fake_api_version = "XXXX-XX-XX"
fake_api_base = "https://faketesturl/"
fake_chunks = ["This is ", "a fake answer."]
fake_answer = "".join(fake_chunks)
token_usage_key_name = "token_usage"

model_list = [
    {
        "model_name": model_group_gpt4,
        "litellm_params": {
            "model": fake_models_names[0],
            "api_key": fake_api_key,
            "api_version": fake_api_version,
            "api_base": fake_api_base,
        },
    },
    {
        "model_name": model_group_to_test,
        "litellm_params": {
            "model": fake_models_names[1],
            "api_key": fake_api_key,
            "api_version": fake_api_version,
            "api_base": fake_api_base,
        },
    },
]


# from https://stackoverflow.com/a/78573267
def aiter_to_iter(it: AsyncIterator) -> Generator:
    "Convert an async iterator into a regular (sync) iterator."
    q_in: queue.SimpleQueue = queue.SimpleQueue()
    q_out: queue.SimpleQueue = queue.SimpleQueue()

    async def threadmain() -> None:
        try:
            # Wait until the sync generator requests an item before continuing
            while q_in.get():
                q_out.put((True, await it.__anext__()))
        except StopAsyncIteration:
            q_out.put((False, None))
        except BaseException as ex:
            q_out.put((False, ex))

    thread = threading.Thread(target=asyncio.run, args=(threadmain(),), daemon=True)
    thread.start()

    try:
        while True:
            q_in.put(True)
            cont, result = q_out.get()
            if cont:
                yield result
            elif result is None:
                break
            else:
                raise result
    finally:
        q_in.put(False)


class FakeCompletion:
    def __init__(self) -> None:
        self.seen_inputs: List[Any] = []

    @staticmethod
    def _get_new_result_and_choices(
        base_result: Dict[str, Any],
    ) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        result = deepcopy(base_result)
        choices = cast(List[Dict[str, Any]], result["choices"])
        return result, choices

    async def _get_fake_results_agenerator(
        self, **kwargs: Any
    ) -> AsyncGenerator[Dict[str, Any], None]:
        from litellm import Usage

        self.seen_inputs.append(kwargs)
        base_result = {
            "choices": [
                {
                    "index": 0,
                }
            ],
            "created": 0,
            "id": "",
            "model": model_group_to_test,
            "object": "chat.completion",
        }
        if kwargs["stream"]:
            for chunk_index in range(0, len(fake_chunks)):
                result, choices = self._get_new_result_and_choices(base_result)
                choice = choices[0]
                choice["delta"] = {
                    "role": "assistant",
                    "content": fake_chunks[chunk_index],
                    "function_call": None,
                }
                choice["finish_reason"] = None
                # no usage here, since no usage from OpenAI API for streaming yet
                # https://community.openai.com/t/usage-info-in-api-responses/18862
                yield result

            result, choices = self._get_new_result_and_choices(base_result)
            choice = choices[0]
            choice["delta"] = {}
            choice["finish_reason"] = "stop"
            # no usage here, since no usage from OpenAI API for streaming yet
            # https://community.openai.com/t/usage-info-in-api-responses/18862
            yield result
        else:
            result, choices = self._get_new_result_and_choices(base_result)
            choice = choices[0]
            choice["message"] = {
                "content": fake_answer,
                "role": "assistant",
            }
            choice["finish_reason"] = "stop"
            result["usage"] = Usage(
                completion_tokens=1, prompt_tokens=2, total_tokens=3
            )
            yield result

    def completion(self, **kwargs: Any) -> Union[List, Dict[str, Any]]:
        agen = self._get_fake_results_agenerator(**kwargs)
        synchronous_iter = aiter_to_iter(agen)
        if kwargs["stream"]:
            results: List[Dict[str, Any]] = []
            while True:
                try:
                    results.append(synchronous_iter.__next__())
                except StopIteration:
                    break
            return results
        else:
            # there is only one result for non-streaming
            return synchronous_iter.__next__()

    async def acompletion(
        self, **kwargs: Any
    ) -> Union[AsyncGenerator[Dict[str, Any], None], Dict[str, Any]]:
        agen = self._get_fake_results_agenerator(**kwargs)
        if kwargs["stream"]:
            return agen
        else:
            # there is only one result for non-streaming
            return await agen.__anext__()

    def check_inputs(self, expected_num_calls: int) -> None:
        assert len(self.seen_inputs) == expected_num_calls
        for kwargs in self.seen_inputs:
            metadata = kwargs["metadata"]

            assert metadata["model_group"] == model_group_to_test

            # LiteLLM router chooses one model name from the model_list
            assert kwargs["model"] in fake_models_names
            assert metadata["deployment"] in fake_models_names

            assert kwargs["api_key"] == fake_api_key
            assert kwargs["api_version"] == fake_api_version
            assert kwargs["api_base"] == fake_api_base


@pytest.fixture
def fake_completion() -> FakeCompletion:
    """Fake AI completion for testing."""
    import litellm

    fake_completion = FakeCompletion()

    # Turn off LiteLLM's built-in telemetry
    litellm.telemetry = False
    litellm.completion = fake_completion.completion
    litellm.acompletion = fake_completion.acompletion
    return fake_completion


@pytest.fixture
def litellm_router() -> Any:
    """LiteLLM router for testing."""
    from litellm import Router

    return Router(model_list=model_list)


@pytest.mark.scheduled
@pytest.mark.enable_socket
def test_litellm_router_call(
    fake_completion: FakeCompletion, litellm_router: Any
) -> None:
    """Test valid call to LiteLLM Router."""
    chat = ChatLiteLLMRouter(router=litellm_router, model_name=model_group_to_test)
    message = HumanMessage(content="Hello")

    response = chat.invoke([message])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == fake_answer
    # no usage check here, since response is only an AIMessage
    fake_completion.check_inputs(expected_num_calls=1)


@pytest.mark.scheduled
@pytest.mark.enable_socket
def test_litellm_router_generate(
    fake_completion: FakeCompletion, litellm_router: Any
) -> None:
    """Test generate method of LiteLLM Router."""
    chat = ChatLiteLLMRouter(router=litellm_router, model_name=model_group_to_test)
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
    assert result.llm_output is not None
    assert result.llm_output[token_usage_key_name] == {
        "completion_tokens": 1,
        "completion_tokens_details": None,
        "prompt_tokens": 2,
        "prompt_tokens_details": None,
        "total_tokens": 3,
    }
    fake_completion.check_inputs(expected_num_calls=1)


@pytest.mark.scheduled
@pytest.mark.enable_socket
def test_litellm_router_streaming(
    fake_completion: FakeCompletion, litellm_router: Any
) -> None:
    """Test streaming tokens from LiteLLM Router."""
    chat = ChatLiteLLMRouter(
        router=litellm_router, model_name=model_group_to_test, streaming=True
    )
    message = HumanMessage(content="Hello")

    response = chat.invoke([message])

    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == fake_answer
    # no usage check here, since response is only an AIMessage
    fake_completion.check_inputs(expected_num_calls=1)


@pytest.mark.scheduled
@pytest.mark.enable_socket
def test_litellm_router_streaming_callback(
    fake_completion: FakeCompletion, litellm_router: Any
) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatLiteLLMRouter(
        router=litellm_router,
        model_name=model_group_to_test,
        streaming=True,
        callbacks=[callback_handler],
        verbose=True,
    )
    message = HumanMessage(content="Write me a sentence with 10 words.")

    response = chat.invoke([message])

    assert callback_handler.llm_streams > 1
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert response.content == fake_answer
    # no usage check here, since response is only an AIMessage
    fake_completion.check_inputs(expected_num_calls=1)


@pytest.mark.scheduled
@pytest.mark.enable_socket
async def test_async_litellm_router(
    fake_completion: FakeCompletion, litellm_router: Any
) -> None:
    """Test async generation."""
    chat = ChatLiteLLMRouter(router=litellm_router, model_name=model_group_to_test)
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
    assert response.llm_output is not None
    assert response.llm_output[token_usage_key_name] == {
        "completion_tokens": 2,
        "completion_tokens_details": None,
        "prompt_tokens": 4,
        "prompt_tokens_details": None,
        "total_tokens": 6,
    }
    fake_completion.check_inputs(expected_num_calls=2)


@pytest.mark.scheduled
@pytest.mark.enable_socket
async def test_async_litellm_router_streaming(
    fake_completion: FakeCompletion, litellm_router: Any
) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    chat = ChatLiteLLMRouter(
        router=litellm_router,
        model_name=model_group_to_test,
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
    fake_completion.check_inputs(expected_num_calls=2)
