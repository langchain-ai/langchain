import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_community.chat_models.llama_edge import LlamaEdgeChatService


@pytest.mark.enable_socket
def test_chat_wasm_service() -> None:
    """This test requires the port 8080 is not occupied."""

    # service url
    service_url = "https://b008-54-186-154-209.ngrok-free.app"

    # create wasm-chat service instance
    chat = LlamaEdgeChatService(service_url=service_url)

    # create message sequence
    system_message = SystemMessage(content="You are an AI assistant")
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [system_message, user_message]

    # chat with wasm-chat service
    response = chat(messages)

    # check response
    assert isinstance(response, AIMessage)
    assert isinstance(response.content, str)
    assert "Paris" in response.content


@pytest.mark.enable_socket
def test_chat_wasm_service_streaming() -> None:
    """This test requires the port 8080 is not occupied."""

    # service url
    service_url = "https://b008-54-186-154-209.ngrok-free.app"

    # create wasm-chat service instance
    chat = LlamaEdgeChatService(service_url=service_url, streaming=True)

    # create message sequence
    user_message = HumanMessage(content="What is the capital of France?")
    messages = [
        user_message,
    ]

    output = ""
    for chunk in chat.stream(messages):
        print(chunk.content, end="", flush=True)
        output += chunk.content

    assert "Paris" in output
