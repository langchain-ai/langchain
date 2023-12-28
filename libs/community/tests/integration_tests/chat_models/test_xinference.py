"""Test Xinference chat."""
import asyncio
import time
from typing import Any, AsyncGenerator, Dict, Generator, Tuple
from unittest.mock import patch

import pytest
import pytest_asyncio

from langchain.callbacks.manager import CallbackManager
from langchain_community.chat_models.xinference import ChatXinference
from langchain.schema import (
    ChatGeneration,
    LLMResult,
)
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler


@pytest.fixture(scope="module")
def event_loop() -> Generator:
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


def test_chat_xinference(setup: Tuple) -> None:
    """Test ChatXinference wrapper."""
    server_url, model_uid = setup
    chat = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_xinference_model(setup: Tuple) -> None:
    """Test ChatXinference wrapper handles model_name."""
    server_url, _ = setup
    with pytest.raises(ValueError):
        ChatXinference(server_url=server_url, model_uid="foo")


def test_chat_xinference_system_message(setup: Tuple) -> None:
    """Test ChatXinference wrapper with system message."""
    server_url, model_uid = setup
    chat = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_xinference_streaming(setup: Tuple) -> None:
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


def test_chat_xinference_streaming_generation_info(setup: Tuple) -> None:
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


def test_chat_xinference_llm_output_contains_model_name(setup: Tuple) -> None:
    """Test llm_output contains model_name."""
    server_url, model_uid = setup
    chat = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_uid


def test_chat_xinference_streaming_llm_output_contains_model_name(setup: Tuple) -> None:
    """Test llm_output contains model_name."""
    server_url, model_uid = setup
    chat = ChatXinference(
        server_url=server_url, model_uid=model_uid, max_tokens=10, stream=True
    )
    message = HumanMessage(content="Hello")
    llm_result = chat.generate([[message]])
    assert llm_result.llm_output is not None
    assert llm_result.llm_output["model_name"] == chat.model_uid


def test_chat_xinference_invalid_streaming_params(setup: Tuple) -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    server_url, model_uid = setup
    with pytest.raises(ValueError):
        ChatXinference(
            server_url=server_url,
            model_uid=model_uid,
            max_tokens=10,
            streaming=True,
            temperature=0,
        )


@pytest.mark.skip(reason="Parallel generation is not supported by ggml.")
@pytest.mark.asyncio
async def test_async_chat_xinference_streaming(setup: Tuple) -> None:
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


def test_chat_xinference_extra_kwargs(setup: Tuple) -> None:
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

    # Test overwrite kwargs
    llm = ChatXinference(
        server_url=server_url,
        model_uid=model_uid,
        max_tokens=10,
        model_kwargs={"temperature": 0},
    )
    assert llm.model_kwargs == {"max_tokens": 10, "temperature": 0}

    mock_called = False

    def _mock_patched_convert(**kwargs: Any) -> Dict:
        nonlocal mock_called
        mock_called = True
        assert kwargs["generate_config"] == {"max_tokens": 11, "temperature": 0}
        return {"prompt": "Hello."}

    with patch(
        "langchain_community.chat_models.xinference._openai_kwargs_to_xinference_kwargs",
        _mock_patched_convert,
    ):
        llm([HumanMessage(content="Hello.")], generate_config={"max_tokens": 11})
        assert mock_called


def test_xinference_streaming(setup: Tuple) -> None:
    """Test streaming tokens from Xinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_astream(setup: Tuple) -> None:
    """Test streaming tokens from Xinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_abatch(setup: Tuple) -> None:
    """Test streaming tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_abatch_tags(setup: Tuple) -> None:
    """Test batch tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


def test_xinference_batch(setup: Tuple) -> None:
    """Test batch tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.asyncio
async def test_xinference_ainvoke(setup: Tuple) -> None:
    """Test invoke tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


def test_xinference_invoke(setup: Tuple) -> None:
    """Test invoke tokens from ChatXinference."""
    server_url, model_uid = setup
    llm = ChatXinference(server_url=server_url, model_uid=model_uid, max_tokens=10)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)


def test_full_example(setup: Tuple) -> None:
    server_url, model_uid = setup
    from langchain.chat_models import ChatXinference
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        HumanMessagePromptTemplate,
        SystemMessagePromptTemplate,
    )
    from langchain.schema import AIMessage, HumanMessage, SystemMessage

    chat = ChatXinference(
        server_url=server_url,
        model_uid=model_uid,  # Use Xinference model UID
        generate_config={
            "max_tokens": 5,
            "temperature": 0,
        },
    )

    messages = [
        SystemMessage(
            content="You are a helpful assistant that translates English to Italian."
        ),
        HumanMessage(
            content="Translate the following sentence from English to Italian: "
            "I love programming."
        ),
    ]
    r = chat(messages)
    assert type(r) is AIMessage
    assert isinstance(r.content, str)
    assert r.content

    template = (
        "You are a helpful assistant that translates "
        "{input_language} to {output_language}."
    )
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # get a chat completion from the formatted messages
    r = chat(
        chat_prompt.format_prompt(
            input_language="English",
            output_language="Italian",
            text="I love programming.",
        ).to_messages(),
        generate_config={"max_tokens": 10},
    )
    assert type(r) is AIMessage
    assert isinstance(r.content, str)
    assert r.content
