import os
from langchain_ibm import ChatWatsonx
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)


WX_APIKEY = os.environ.get("WATSONX_APIKEY", "")
WX_PROJECT_ID = os.environ.get("WATSONX_PROJECT_ID", "")

URL = "https://us-south.ml.cloud.ibm.com"
MODEL_ID = "google/flan-t5-xl"

DOCUMENTS = ["What is a generative ai?", "What is a loan and how does it works?"]


def test_01_generate_chat() -> None:
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID
    )
    messages = [
        ("system", "You are a helpful assistant that translates English to French."),
        ("human", "Translate this sentence from English to French. I love programming."),
    ]
    response = chat.invoke(messages)
    assert response


def test_02_generate_chat() -> None:
    """Test ChatOpenAI wrapper with generate."""
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID
    )
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert response


def test_03_generate_chat() -> None:
    """Test ChatOpenAI wrapper with system message."""
    chat = ChatWatsonx(
        model_id=MODEL_ID, url=URL, project_id=WX_PROJECT_ID
    )
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat.invoke([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)
