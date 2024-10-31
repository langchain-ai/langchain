"""Test Nebius AI Studio chat model."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_community.chat_models.nebius_ai_studio import ChatNebiusAIStudio


def test_chat_nebius_ai_studio_initialization():
    """Test initialization of chat model."""
    chat = ChatNebiusAIStudio(
        nebius_api_key="test-api-key", model="meta-llama/Meta-Llama-3.1-70B-Instruct"
    )
    assert chat.model_name == "meta-llama/Meta-Llama-3.1-70B-Instruct"
    assert chat._llm_type == "nebius-ai-studio-chat"
    assert chat.nebius_api_key.get_secret_value() == "test-api-key"
