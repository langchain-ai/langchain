"""Test ChatGoogleVertexAI chat model."""
from typing import Optional, cast

import pytest
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, LLMResult

from langchain_google_vertexai.chat_models import ChatVertexAI

model_names_to_test = [None, "codechat-bison", "chat-bison", "gemini-pro"]


@pytest.mark.parametrize("model_name", model_names_to_test)
def test_initialization(model_name: Optional[str]) -> None:
    """Test chat model initialization."""
    if model_name:
        model = ChatVertexAI(model_name=model_name)
    else:
        model = ChatVertexAI()
    assert model._llm_type == "vertexai"
    try:
        assert model.model_name == model.client._model_id
    except AttributeError:
        assert model.model_name == model.client._model_name.split("/")[-1]


@pytest.mark.parametrize("model_name", model_names_to_test)
def test_vertexai_single_call(model_name: Optional[str]) -> None:
    if model_name:
        model = ChatVertexAI(model_name=model_name)
    else:
        model = ChatVertexAI()
    message = HumanMessage(content="Hello")
    response = model([message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


# mark xfail because Vertex API randomly doesn't respect
# the n/candidate_count parameter
@pytest.mark.xfail
def test_candidates() -> None:
    model = ChatVertexAI(model_name="chat-bison@001", temperature=0.3, n=2)
    message = HumanMessage(content="Hello")
    response = model.generate(messages=[[message]])
    assert isinstance(response, LLMResult)
    assert len(response.generations) == 1
    assert len(response.generations[0]) == 2


@pytest.mark.parametrize("model_name", ["chat-bison@001", "gemini-pro"])
async def test_vertexai_agenerate(model_name: str) -> None:
    model = ChatVertexAI(temperature=0, model_name=model_name)
    message = HumanMessage(content="Hello")
    response = await model.agenerate([[message]])
    assert isinstance(response, LLMResult)
    assert isinstance(response.generations[0][0].message, AIMessage)  # type: ignore

    sync_response = model.generate([[message]])
    sync_generation = cast(ChatGeneration, sync_response.generations[0][0])
    async_generation = cast(ChatGeneration, response.generations[0][0])

    # assert some properties to make debugging easier

    # xfail: this is not equivalent with temp=0 right now
    # assert sync_generation.message.content == async_generation.message.content
    assert sync_generation.generation_info == async_generation.generation_info

    # xfail: content is not same right now
    # assert sync_generation == async_generation


@pytest.mark.parametrize("model_name", ["chat-bison@001", "gemini-pro"])
def test_vertexai_stream(model_name: str) -> None:
    model = ChatVertexAI(temperature=0, model_name=model_name)
    message = HumanMessage(content="Hello")

    sync_response = model.stream([message])
    for chunk in sync_response:
        assert isinstance(chunk, AIMessageChunk)


def test_vertexai_single_call_with_context() -> None:
    model = ChatVertexAI()
    raw_context = (
        "My name is Ned. You are my personal assistant. My favorite movies "
        "are Lord of the Rings and Hobbit."
    )
    question = (
        "Hello, could you recommend a good movie for me to watch this evening, please?"
    )
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = model([context, message])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_multimodal() -> None:
    llm = ChatVertexAI(model_name="gemini-pro-vision")
    gcs_url = (
        "gs://cloud-samples-data/generative-ai/image/"
        "320px-Felis_catus-cat_on_snow.jpg"
    )
    image_message = {
        "type": "image_url",
        "image_url": {"url": gcs_url},
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    message = HumanMessage(content=[text_message, image_message])
    output = llm([message])
    assert isinstance(output.content, str)


@pytest.mark.xfail(reason="problem on vertex side")
def test_multimodal_history() -> None:
    llm = ChatVertexAI(model_name="gemini-pro-vision")
    gcs_url = (
        "gs://cloud-samples-data/generative-ai/image/"
        "320px-Felis_catus-cat_on_snow.jpg"
    )
    image_message = {
        "type": "image_url",
        "image_url": {"url": gcs_url},
    }
    text_message = {
        "type": "text",
        "text": "What is shown in this image?",
    }
    message1 = HumanMessage(content=[text_message, image_message])
    message2 = AIMessage(
        content=(
            "This is a picture of a cat in the snow. The cat is a tabby cat, which is "
            "a type of cat with a striped coat. The cat is standing in the snow, and "
            "its fur is covered in snow."
        )
    )
    message3 = HumanMessage(content="What time of day is it?")
    response = llm([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_vertexai_single_call_with_examples() -> None:
    model = ChatVertexAI()
    raw_context = "My name is Ned. You are my personal assistant."
    question = "2+2"
    text_question, text_answer = "4+4", "8"
    inp = HumanMessage(content=text_question)
    output = AIMessage(content=text_answer)
    context = SystemMessage(content=raw_context)
    message = HumanMessage(content=question)
    response = model([context, message], examples=[inp, output])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


@pytest.mark.parametrize("model_name", model_names_to_test)
def test_vertexai_single_call_with_history(model_name: Optional[str]) -> None:
    if model_name:
        model = ChatVertexAI(model_name=model_name)
    else:
        model = ChatVertexAI()
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)


def test_vertexai_single_call_fails_no_message() -> None:
    chat = ChatVertexAI()
    with pytest.raises(ValueError) as exc_info:
        _ = chat([])
    assert (
        str(exc_info.value)
        == "You should provide at least one message to start the chat!"
    )


@pytest.mark.parametrize("model_name", ["gemini-pro"])
def test_chat_vertexai_gemini_system_message_error(model_name: str) -> None:
    model = ChatVertexAI(model_name=model_name)
    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content="You're supposed to answer math questions.")
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    with pytest.raises(ValueError):
        model([system_message, message1, message2, message3])


@pytest.mark.parametrize("model_name", model_names_to_test)
def test_chat_vertexai_system_message(model_name: Optional[str]) -> None:
    if model_name:
        model = ChatVertexAI(
            model_name=model_name, convert_system_message_to_human=True
        )
    else:
        model = ChatVertexAI()

    text_question1, text_answer1 = "How much is 2+2?", "4"
    text_question2 = "How much is 3+3?"
    system_message = SystemMessage(content="You're supposed to answer math questions.")
    message1 = HumanMessage(content=text_question1)
    message2 = AIMessage(content=text_answer1)
    message3 = HumanMessage(content=text_question2)
    response = model([system_message, message1, message2, message3])
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
