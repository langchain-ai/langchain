"""Test Xinference chat."""
import asyncio
import time
from typing import Any, AsyncGenerator, Tuple

import pytest
import pytest_asyncio

from langchain.callbacks.manager import CallbackManager
from langchain.chat_models.xinference import ChatXinference
from langchain.schema import (
    ChatGeneration,
    ChatResult,
    LLMResult,
)
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture(scope="module")
async def setup() -> AsyncGenerator[Tuple[str, str], None]:
    import xoscar as xo
    from xinference.client import RESTfulClient
    from xinference.deploy.supervisor import start_supervisor_components
    from xinference.deploy.utils import create_worker_actor_pool
    from xinference.deploy.worker import start_worker_components

    pool = await create_worker_actor_pool(
        f"test://127.0.0.1:{xo.utils.get_next_port()}"
    )
    print(f"Pool running on localhost:{pool.external_address}")

    endpoint = await start_supervisor_components(
        pool.external_address, "127.0.0.1", xo.utils.get_next_port()
    )
    await start_worker_components(
        address=pool.external_address,
        supervisor_address=pool.external_address,
        main_pool=pool,
    )

    # wait for the api.
    time.sleep(3)
    async with pool:
        client = RESTfulClient(endpoint)

        model_uid = client.launch_model(
            model_name="vicuna-v1.3",
            model_size_in_billions=7,
            model_format="ggmlv3",
            quantization="q4_0",
        )

        yield endpoint, model_uid


def test_chat_xinference(setup) -> None:
    """Test ChatXinference wrapper."""
    server_url, model_uid = setup
    chat = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_xinference_model(setup) -> None:
    """Test ChatXinference wrapper handles model_name."""
    server_url, _ = setup
    with pytest.raises(ValueError):
        ChatXinference(server_url=server_url, model_uid="foo")


def test_chat_xinference_system_message(setup) -> None:
    """Test ChatXinference wrapper with system message."""
    server_url, model_uid = setup
    chat = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


@pytest.mark.skip(reason="n is not supported by Xinference chat.")
def test_chat_xinference_generate(setup) -> None:
    """Test ChatXinference wrapper with generate."""
    server_url, model_uid = setup
    chat = ChatXinference(
        server_url=server_url, model_uid=model_uid, max_tokens=10, n=2
    )
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.skip(reason="n is not supported by Xinference chat.")
def test_chat_xinference_multiple_completions(setup) -> None:
    """Test ChatXinference wrapper with multiple completions."""
    server_url, model_uid = setup
    chat = ChatXinference(
        server_url=server_url, model_uid=model_uid, max_tokens=10, n=5
    )
    message = HumanMessage(content="Hello")
    response = chat._generate([message])
    assert isinstance(response, ChatResult)
    assert len(response.generations) == 5
    for generation in response.generations:
        assert isinstance(generation.message, BaseMessage)
        assert isinstance(generation.message.content, str)


def test_chat_xinference_streaming(setup) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    server_url, model_uid = setup
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatXinference(
        server_url=server_url,
        model_uid=model_uid,
        max_tokens=10,
        stream=True,
        temperature=0,
        callback_manager=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)


def test_chat_xinference_streaming_generation_info(setup) -> None:
    """Test that generation info is preserved when streaming."""
    server_url, model_uid = setup

    class _FakeCallback(FakeCallbackHandler):
        saved_things: dict = {}

        def on_llm_end(
            self,
            *args: Any,
            **kwargs: Any,
        ) -> Any:
            # Save the generation
            self.saved_things["generation"] = args[0]

    callback = _FakeCallback()
    callback_manager = CallbackManager([callback])
    chat = ChatXinference(
        server_url=server_url,
        model_uid=model_uid,
        max_tokens=2,
        temperature=0,
        callback_manager=callback_manager,
    )
    list(chat.stream("hi"))
    generation = callback.saved_things["generation"]
    # `Hello!` is two tokens, assert that that is what is returned
    assert "Hello!" in generation.generations[0][0].text


def test_chat_xinference_llm_output_contains_model_name(setup) -> None:
    """Test llm_output contains model_name."""
    server_url, model_uid = setup
    chat = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_uid


def test_chat_xinference_streaming_llm_output_contains_model_name(setup) -> None:
    """Test llm_output contains model_name."""
    server_url, model_uid = setup
    chat = ChatXinference(
        server_url=server_url, model_uid=model_uid, max_tokens=10, stream=True
    )
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_uid


def test_chat_xinference_invalid_streaming_params(setup) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    server_url, model_uid = setup
    with pytest.raises(ValueError):
        ChatXinference(
            server_url=server_url,
            model_uid=model_uid,
            max_tokens=10,
            streaming=True,
            temperature=0,
            n=5,
        )


@pytest.mark.skip(reason="n is not supported by Xinference chat.")
@pytest.mark.asyncio
async def test_async_chat_xinference(setup) -> None:
    """Test async generation."""
    server_url, model_uid = setup
    chat = ChatXinference(
        server_url=server_url, model_uid=model_uid, max_tokens=10, n=2
    )
    message = HumanMessage(content="Hello")
    response = await chat.agenerate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 2
    for generations in response.generations:
        assert len(generations) == 2
        for generation in generations:
            assert isinstance(generation, ChatGeneration)
            assert isinstance(generation.text, str)
            assert generation.text == generation.message.content


@pytest.mark.skip(reason="Parallel generation is not supported by ggml.")
@pytest.mark.asyncio
async def test_async_chat_xinference_streaming(setup) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    server_url, model_uid = setup
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = ChatXinference(
        server_url=server_url,
        model_uid=model_uid,
        max_tokens=10,
        streaming=True,
        temperature=0,
        callback_manager=callback_manager,
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
            assert generation.text == generation.message.content


def test_chat_xinference_extra_kwargs(setup) -> None:
    """Test extra kwargs to chat xinference."""
    server_url, model_uid = setup
    # Check that foo is saved in extra_kwargs.
    llm = ChatXinference(
        server_url=server_url, model_uid=model_uid, foo=3, max_tokens=10
    )
    assert llm.model_kwargs == {"foo": 3, "max_tokens": 10}

    # Test that if extra_kwargs are provided, they are added to it.
    llm = ChatXinference(
        server_url=server_url, model_uid=model_uid, foo=3, model_kwargs={"bar": 2}
    )
    assert llm.model_kwargs == {"foo": 3, "bar": 2}

    # Test that if provided twice it errors
    with pytest.raises(ValueError):
        ChatXinference(
            server_url=server_url, model_uid=model_uid, foo=3, model_kwargs={"foo": 2}
        )


def test_xinference_streaming(setup) -> None:
    """Test streaming tokens from Xinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_astream(setup) -> None:
    """Test streaming tokens from Xinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_abatch(setup) -> None:
    """Test streaming tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_abatch_tags(setup) -> None:
    """Test batch tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_xinference_batch(setup) -> None:
    """Test batch tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_ainvoke(setup) -> None:
    """Test invoke tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_xinference_invoke(setup) -> None:
    """Test invoke tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
