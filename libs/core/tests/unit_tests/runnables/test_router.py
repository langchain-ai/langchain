import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.tiered_router import TieredSemanticRouter

def test_tiered_router_routing_logic():
    # Setup Mock Models with spec
    primary = MagicMock(spec=BaseChatModel) # Fix: add spec
    primary.invoke.return_value = AIMessage(content="small response")
    
    fallback = MagicMock(spec=BaseChatModel) # Fix: add spec
    fallback.invoke.return_value = AIMessage(content="large response")

    # Initialize Router - Pydantic will now accept these mocks
    router = TieredSemanticRouter(primary=primary, fallback=fallback, threshold=0.5)

    # Test Case 1: Simple input should go to Primary
    router.invoke("Hi")
    primary.invoke.assert_called_once()
    fallback.invoke.assert_not_called()

    primary.invoke.reset_mock()

    # Test Case 2: Complex input should go to Fallback
    complex_input = "Analyze the following code for security vulnerabilities" + "a" * 500
    router.invoke(complex_input)
    fallback.invoke.assert_called_once()