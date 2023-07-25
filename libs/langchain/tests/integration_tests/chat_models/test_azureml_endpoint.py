"""Test AzureML Endpoint wrapper."""
import pytest

from langchain.chat_models.azureml_endpoint import (
    LlamaContentFormatter,
    AzureMLChatOnlineEndpoint
)
from langchain.schema import ChatResult, HumanMessage, SystemMessage


def test_llama_call() -> None:
    """Test valid call to Open Source Foundation Model."""
    llm = AzureMLChatOnlineEndpoint(content_formatter=LlamaContentFormatter())
    output = llm(messages=[
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content="Why is the sky blue?")
            ]
        )
    assert isinstance(output, ChatResult)
