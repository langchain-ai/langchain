import pytest
from unittest.mock import MagicMock
from langchain_core.messages import AIMessage
from langchain_core.language_models import BaseChatModel
from langchain_core.runnables.tiered_router import TieredSemanticRouter

@pytest.fixture
def mock_models():
    primary = MagicMock(spec=BaseChatModel)
    primary.invoke.return_value = AIMessage(content="primary")
    primary.ainvoke.return_value = AIMessage(content="primary_async")
    
    fallback = MagicMock(spec=BaseChatModel)
    fallback.invoke.return_value = AIMessage(content="fallback")
    fallback.ainvoke.return_value = AIMessage(content="fallback_async")
    return primary, fallback

def test_tiered_router_routing_logic(mock_models):
    primary, fallback = mock_models
    router = TieredSemanticRouter(primary=primary, fallback=fallback, threshold=0.5)

    # 1. Test Primary Route
    router.invoke("Hi")
    primary.invoke.assert_called_once()
    fallback.invoke.assert_not_called()

    primary.reset_mock()
    fallback.reset_mock()

    # 2. Test Fallback Route
    complex_input = "Analyze this code " + "x" * 600
    router.invoke(complex_input)
    fallback.invoke.assert_called_once()
    primary.invoke.assert_not_called()

@pytest.mark.asyncio
async def test_tiered_router_async_logic(mock_models):
    primary, fallback = mock_models
    router = TieredSemanticRouter(primary=primary, fallback=fallback, threshold=0.5)

    # Verify ainvoke works correctly
    await router.ainvoke("Hi")
    primary.ainvoke.assert_called_once()

def test_tiered_router_config_propagation(mock_models):
    """Ensure tags and metadata are passed down to the underlying models."""
    primary, fallback = mock_models
    router = TieredSemanticRouter(primary=primary, fallback=fallback, threshold=0.5)
    
    config = {"tags": ["test-tag"], "metadata": {"user_id": "123"}}
    router.invoke("Hi", config=config)
    
    # Check if the config was passed to the primary model's invoke
    args, kwargs = primary.invoke.call_args
    assert kwargs["config"]["tags"] == ["test-tag"]