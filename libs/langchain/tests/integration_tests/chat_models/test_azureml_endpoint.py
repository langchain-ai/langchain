"""Test AzureML Endpoint wrapper."""
import os

import pytest

from langchain.chat_models.azureml_endpoint import (
    LlamaContentFormatter,
    AzureMLChatOnlineEndpoint
)
from langchain.schema import BaseMessage, HumanMessage


def test_llama_call() -> None:
    """Test valid call to Open Source Foundation Model."""
    llm = AzureMLChatOnlineEndpoint(
        endpoint_api_key=os.getenv("AZUREML_ENDPOINT_API_KEY"),
        endpoint_url=os.getenv("AZUREML_ENDPOINT_URL"),
        content_formatter=LlamaContentFormatter()
    )
    output = llm(messages=[
                HumanMessage(content="Why is the sky blue?")
            ]
        )
    assert isinstance(output, BaseMessage)
