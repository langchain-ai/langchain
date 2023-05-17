"""Test Google PaLM Chat API wrapper.

To use you must have the google-cloud-aiplatform Python package installed and
either:

    1. Have credentials configured for your environment (gcloud, workload identity, etc...)
    2. Pass your service account key json using the google_application_credentials kwarg to the ChatGoogle
        constructor.

    *see: https://cloud.google.com/docs/authentication/application-default-credentials#GAC
"""

import pytest

from langchain.llms.vertex_ai_palm import ChatGoogleCloudVertexAIPalm
from langchain.schema import (
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    LLMResult,
    SystemMessage,
)

try:
    from vertexai.preview.language_models import ChatModel # noqa: F401

    vertexai_installed = True
except ImportError:
    vertexai_installed = False


@pytest.mark.skipif(not vertexai_installed, reason="google-cloud-aiplatform>=1.25.0 package not installed")
def test_chat_google_palm() -> None:
    """Test Google PaLM Vertex AI Chat API wrapper."""
    chat = ChatGoogleCloudVertexAIPalm()
    message = HumanMessage(content="Hello")
    response = chat([message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

@pytest.mark.skipif(not vertexai_installed, reason="google-cloud-aiplatform>=1.25.0 package not installed")
def test_chat_google_palm_system_message() -> None:
    """Test Google PaLM Chat Vertex AI API wrapper with system message."""
    chat = ChatGoogleCloudVertexAIPalm()
    system_message = SystemMessage(content="You are to chat with the user.")
    human_message = HumanMessage(content="Hello")
    response = chat([system_message, human_message])
    assert isinstance(response, BaseMessage)
    assert isinstance(response.content, str)

@pytest.mark.skipif(not vertexai_installed, reason="google-cloud-aiplatform>=1.25.0 package not installed")
def test_chat_google_palm_generate() -> None:
    """Test Google PaLM Chat API wrapper with generate."""
    chat = ChatGoogleCloudVertexAIPalm(temperature=1.0)
    message = HumanMessage(content="Hello")
    response = chat.generate([[message], [message]])
    assert isinstance(response, LLMResult)
    assert isinstance(response.content, str)