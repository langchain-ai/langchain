"""Test LlamaChat wrapper."""

# from typing import Any, List, Optional, Union

import pytest

from langchain.callbacks.base import AsyncCallbackHandler
from langchain.callbacks.manager import CallbackManager

# from langchain.chat_models.aiplay import LlamaChat
from langchain.chat_models.aiplay import AIPlayChat, LlamaChat  
# from langchain.output_parsers.aiplay_functions import JsonOutputFunctionsParser
# from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
# from langchain.schema import (
#     ChatGeneration,
#     ChatResult,
#     LLMResult,
# )
from langchain.schema.messages import BaseMessage, HumanMessage, SystemMessage
# from langchain.schema.output import ChatGenerationChunk, GenerationChunk
from tests.unit_tests.callbacks.fake_callback_handler import FakeCallbackHandler

######################################################################
## NOTE: Commandeering ChatOpenAI tests to see what we're missing. 
## Some tests are commented out to represent features we'd like 
## to support later or are otherwise waiting to have dialog on.
## Interested parties can try to add support or discard tests as time permits. 

@pytest.mark.scheduled
def test_chat_aiplay() -> None:
    """Test AIPlayChat wrapper."""
    chat = LlamaChat(
        temperature=0.7,
        # base_url=None,
        # organization=None,
        # aiplay_proxy=None,
        # timeout=10.0,
        # max_retries=3,
        # http_client=None,
        # n=1,
        # max_tokens=60,
        # default_headers=None,
        # default_query=None,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_chat_aiplay_model() -> None:
    """Test LlamaChat wrapper handles model_name."""
    # chat = LlamaChat(model="foo")
    # assert chat.model_name == "foo"
    # chat = LlamaChat(model_name="bar")
    # assert chat.model_name == "bar"
    chat = AIPlayChat(model_name="mistral")
    assert chat.model_name == "mistral"

    chat = LlamaChat(model="mistral")
    assert chat.model_name == "mistral"


def test_chat_aiplay_system_message() -> None:
    """Test LlamaChat wrapper with system message."""
    chat = LlamaChat(max_tokens=36)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


## TODO: Not sure if we want to support the n syntax. Trash or keep test
# @pytest.mark.scheduled
# def test_chat_aiplay_generate() -> None:
#     """Test LlamaChat wrapper with generate."""
#     chat = LlamaChat(max_tokens=60, n=2)
#     message = HumanMessage(content="Hello")
#     response = chat.generate([[message], [message]])
#     assert isinstance(response, LLMResult)
#     assert len(response.generations) == 2
#     assert response.llm_output
#     assert "system_fingerprint" in response.llm_output
#     for generations in response.generations:
#         assert len(generations) == 2
#         for generation in generations:
#             assert isinstance(generation, ChatGeneration)
#             assert isinstance(generation.text, str)
#             assert generation.text == generation.message.content


# @pytest.mark.scheduled
# def test_chat_aiplay_multiple_completions() -> None:
#     """Test LlamaChat wrapper with multiple completions."""
#     chat = LlamaChat(max_tokens=60, n=5)
#     message = HumanMessage(content="Hello")
#     response = chat._generate([message])
#     assert isinstance(response, ChatResult)
#     assert len(response.generations) == 5
#     for generation in response.generations:
#         assert isinstance(generation.message, BaseMessage)
#         assert isinstance(generation.message.content, str)

######################################################################
## TODO: Callback handling not supported yet...

@pytest.mark.scheduled
def test_chat_aiplay_streaming() -> None:
    """Test that streaming correctly invokes on_llm_new_token callback."""
    callback_handler = FakeCallbackHandler()
    callback_manager = CallbackManager([callback_handler])
    chat = AIPlayChat(
        max_tokens=36,
        streaming=True,
        temperature=0.1,
        callbacks=callback_manager,
        verbose=True,
    )
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert callback_handler.llm_streams > 0
    assert isinstance(response, BaseMessage)

######################################################################
## We do not support this yet. Training dependent, and no token-limits

# @pytest.mark.scheduled
# def test_chat_aiplay_streaming_generation_info() -> None:
#     """Test that generation info is preserved when streaming."""

#     class _FakeCallback(FakeCallbackHandler):
#         saved_things: dict = {}

#         def on_llm_end(
#             self,
#             *args: Any,
#             **kwargs: Any,
#         ) -> Any:
#             # Save the generation
#             self.saved_things["generation"] = args[0]

#     callback = _FakeCallback()
#     callback_manager = CallbackManager([callback])
#     chat = LlamaChat(
#         max_tokens=2,
#         temperature=0,
#         callback_manager=callback_manager,
#     )
#     list(chat.stream("hi"))
#     generation = callback.saved_things["generation"]
#     # `Hello!` is two tokens, assert that that is what is returned
#     assert generation.generations[0][0].text == "Hello!"

######################################################################
## Tests do not pass yet since we are not implementing llm_output meta

# def test_chat_aiplay_llm_output_contains_model_name() -> None:
#     """Test llm_output contains model_name."""
#     chat = LlamaChat(max_tokens=60)
#     message = HumanMessage(content="Hello")
#     llm_result = chat.generate([[message]])
#     assert llm_result.llm_output is not None
#     assert llm_result.llm_output["model_name"] == chat.model_name


# def test_chat_aiplay_streaming_llm_output_contains_model_name() -> None:
#     """Test llm_output contains model_name."""
#     chat = LlamaChat(max_tokens=60, streaming=True)
#     message = HumanMessage(content="Hello")
#     llm_result = chat.generate([[message]])
#     assert llm_result.llm_output is not None
#     assert llm_result.llm_output["model_name"] == chat.model_name

######################################################################

# def test_chat_aiplay_invalid_streaming_params() -> None:
#     """Test that streaming correctly invokes on_llm_new_token callback."""
#     with pytest.raises(ValueError):
#         LlamaChat(
#             max_tokens=60,
#             streaming=True,
#             temperature=0,
#             n=5,
#         )

######################################################################
## We probbaly don't want to support n and max_tokens at the moment 
## since this is AI Playground, so no care for output length

# @pytest.mark.scheduled
# @pytest.mark.asyncio
# async def test_async_chat_aiplay() -> None:
#     """Test async generation."""
#     chat = LlamaChat(max_tokens=60, n=2)
#     message = HumanMessage(content="Hello")
#     response = await chat.agenerate([[message], [message]])
#     assert isinstance(response, LLMResult)
#     assert len(response.generations) == 2
#     assert response.llm_output
#     assert "system_fingerprint" in response.llm_output
#     for generations in response.generations:
#         assert len(generations) == 2
#         for generation in generations:
#             assert isinstance(generation, ChatGeneration)
#             assert isinstance(generation.text, str)
#             assert generation.text == generation.message.content

######################################################################
## Callback managers not supported yet

# @pytest.mark.scheduled
# @pytest.mark.asyncio
# async def test_async_chat_aiplay_streaming() -> None:
#     """Test that streaming correctly invokes on_llm_new_token callback."""
#     callback_handler = FakeCallbackHandler()
#     callback_manager = CallbackManager([callback_handler])
#     chat = LlamaChat(
#         max_tokens=60,
#         streaming=True,
#         temperature=0,
#         callback_manager=callback_manager,
#         verbose=True,
#     )
#     message = HumanMessage(content="Hello")
#     response = await chat.agenerate([[message], [message]])
#     assert callback_handler.llm_streams > 0
#     assert isinstance(response, LLMResult)
#     assert len(response.generations) == 2
#     for generations in response.generations:
#         assert len(generations) == 1
#         for generation in generations:
#             assert isinstance(generation, ChatGeneration)
#             assert isinstance(generation.text, str)
#             assert generation.text == generation.message.content

######################################################################
## Would be nice to support functions soon

# @pytest.mark.scheduled
# @pytest.mark.asyncio
# async def test_async_chat_aiplay_streaming_with_function() -> None:
#     """Test LlamaChat wrapper with multiple completions."""

#     class MyCustomAsyncHandler(AsyncCallbackHandler):
#         def __init__(self) -> None:
#             super().__init__()
#             self._captured_tokens: List[str] = []
#             self._captured_chunks: List[
#                 Optional[Union[ChatGenerationChunk, GenerationChunk]]
#             ] = []

#         def on_llm_new_token(
#             self,
#             token: str,
#             *,
#             chunk: Optional[Union[ChatGenerationChunk, GenerationChunk]] = None,
#             **kwargs: Any,
#         ) -> Any:
#             self._captured_tokens.append(token)
#             self._captured_chunks.append(chunk)

#     json_schema = {
#         "title": "Person",
#         "description": "Identifying information about a person.",
#         "type": "object",
#         "properties": {
#             "name": {
#                 "title": "Name",
#                 "description": "The person's name",
#                 "type": "string",
#             },
#             "age": {
#                 "title": "Age",
#                 "description": "The person's age",
#                 "type": "integer",
#             },
#             "fav_food": {
#                 "title": "Fav Food",
#                 "description": "The person's favorite food",
#                 "type": "string",
#             },
#         },
#         "required": ["name", "age"],
#     }

#     callback_handler = MyCustomAsyncHandler()
#     callback_manager = CallbackManager([callback_handler])

#     chat = LlamaChat(
#         # max_tokens=60,
#         n=1,
#         callback_manager=callback_manager,
#         streaming=True,
#     )

#     prompt_msgs = [
#         SystemMessage(
#             content="You are a world class algorithm for "
#             "extracting information in structured formats."
#         ),
#         HumanMessage(
#             content="Use the given format to extract "
#             "information from the following input:"
#         ),
#         HumanMessagePromptTemplate.from_template("{input}"),
#         HumanMessage(content="Tips: Make sure to answer in the correct format"),
#     ]
#     prompt = ChatPromptTemplate(messages=prompt_msgs)

#     function: Any = {
#         "name": "output_formatter",
#         "description": (
#             "Output formatter. Should always be used to format your response to the"
#             " user."
#         ),
#         "parameters": json_schema,
#     }
#     chain = create_aiplay_fn_chain(
#         [function],
#         chat,
#         prompt,
#         output_parser=None,
#     )

#     message = HumanMessage(content="Sally is 13 years old")
#     response = await chain.agenerate([{"input": message}])

#     assert isinstance(response, LLMResult)
#     assert len(response.generations) == 1
#     for generations in response.generations:
#         assert len(generations) == 1
#         for generation in generations:
#             assert isinstance(generation, ChatGeneration)
#             assert isinstance(generation.text, str)
#             assert generation.text == generation.message.content
#     assert len(callback_handler._captured_tokens) > 0
#     assert len(callback_handler._captured_chunks) > 0
#     assert all([chunk is not None for chunk in callback_handler._captured_chunks])

# @pytest.mark.scheduled
# @pytest.mark.asyncio
# async def test_async_chat_aiplay_bind_functions() -> None:
#     """Test LlamaChat wrapper with multiple completions."""

#     class Person(BaseModel):
#         """Identifying information about a person."""

#         name: str = Field(..., title="Name", description="The person's name")
#         age: int = Field(..., title="Age", description="The person's age")
#         fav_food: Optional[str] = Field(
#             default=None, title="Fav Food", description="The person's favorite food"
#         )

#     chat = LlamaChat(
#         # max_tokens=30,
#         n=1,
#         streaming=True,
#     ).bind_functions(functions=[Person], function_call="Person")

#     prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "Use the provided Person function"),
#             ("user", "{input}"),
#         ]
#     )

#     chain = prompt | chat | JsonOutputFunctionsParser(args_only=True)

#     message = HumanMessage(content="Sally is 13 years old")
#     response = await chain.abatch([{"input": message}])

#     assert isinstance(response, list)
#     assert len(response) == 1
#     for generation in response:
#         assert isinstance(generation, dict)
#         assert "name" in generation
#         assert "age" in generation

######################################################################

## Not completely consistent with our syntax. Do we want to support this?

# def test_chat_aiplay_extra_kwargs() -> None:
#     """Test extra kwargs to chat aiplay."""
#     # Check that foo is saved in extra_kwargs.
#     llm = LlamaChat(foo=3, max_tokens=60)
#     assert llm.max_tokens == 10
#     assert llm.model_kwargs == {"foo": 3}

#     # Test that if extra_kwargs are provided, they are added to it.
#     llm = LlamaChat(foo=3, model_kwargs={"bar": 2})
#     assert llm.model_kwargs == {"foo": 3, "bar": 2}

#     # Test that if provided twice it errors
#     with pytest.raises(ValueError):
#         LlamaChat(foo=3, model_kwargs={"foo": 2})

#     # Test that if explicit param is specified in kwargs it errors
#     with pytest.raises(ValueError):
#         LlamaChat(model_kwargs={"temperature": 0.2})

#     # Test that "model" cannot be specified in kwargs
#     with pytest.raises(ValueError):
#         LlamaChat(model_kwargs={"model": "text-davinci-003"})

######################################################################

@pytest.mark.scheduled
def test_aiplay_streaming() -> None:
    """Test streaming tokens from aiplay."""
    llm = LlamaChat(max_tokens=36)

    for token in llm.stream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_astream() -> None:
    """Test streaming tokens from aiplay."""
    llm = LlamaChat(max_tokens=35)

    async for token in llm.astream("I'm Pickle Rick"):
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_abatch() -> None:
    """Test streaming tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=36)

    result = await llm.abatch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_abatch_tags() -> None:
    """Test batch tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=55)

    result = await llm.abatch(
        ["I'm Pickle Rick", "I'm not Pickle Rick"], config={"tags": ["foo"]}
    )
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
def test_aiplay_batch() -> None:
    """Test batch tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=60)

    result = llm.batch(["I'm Pickle Rick", "I'm not Pickle Rick"])
    for token in result:
        assert isinstance(token.content, str)


@pytest.mark.scheduled
@pytest.mark.asyncio
async def test_aiplay_ainvoke() -> None:
    """Test invoke tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=60)

    result = await llm.ainvoke("I'm Pickle Rick", config={"tags": ["foo"]})
    assert isinstance(result.content, str)


@pytest.mark.scheduled
def test_aiplay_invoke() -> None:
    """Test invoke tokens from LlamaChat."""
    llm = LlamaChat(max_tokens=60)

    result = llm.invoke("I'm Pickle Rick", config=dict(tags=["foo"]))
    assert isinstance(result.content, str)
