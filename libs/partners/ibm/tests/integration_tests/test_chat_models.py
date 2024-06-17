import os

from ibm_watsonx_ai.metanames import GenTextParamsMetaNames  # type: ignore
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

from langchain_ibm import ChatWatsonx

WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "ibm/granite-13b-chat-v2"

DOCUMENTS = ["What is a generative ai?", "What is a loan and how does it works?"]


def test_01_generate_chat() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    messages = [
        ("system", "You are a helpful assistant that translates English to French."),
        (
            "human",
            "Translate this sentence from English to French. I love programming.",
        ),
    ]
    response = chat.invoke(messages)
    assert response


def test_02_generate_chat_with_few_inputs() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert response


def test_03_generate_chat_with_few_various_inputs() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)


def test_05_generate_chat_with_stream() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    response = chat.stream("What's the weather in san francisco")
    for chunk in response:
        assert isinstance(chunk.content, str)


def test_10_chaining() -> None:
    chat = ChatWatsonx(model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that "
                "translates {input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat

    response = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    assert response


def test_11_chaining_with_params() -> None:
    parameters = {
        GenTextParamsMetaNames.DECODING_METHOD: "sample",
        GenTextParamsMetaNames.MIN_NEW_TOKENS: 5,
        GenTextParamsMetaNames.MAX_NEW_TOKENS: 10,
    }
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID, params=parameters
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant that translates "
                "{input_language} to {output_language}.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | chat

    response = chain.invoke(
        {
            "input_language": "English",
            "output_language": "German",
            "input": "I love programming.",
        }
    )
    assert response
