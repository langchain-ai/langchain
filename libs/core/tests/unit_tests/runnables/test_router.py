from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables.tiered_router import TieredSemanticRouter

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


@pytest.fixture
def mock_models() -> tuple[MagicMock, MagicMock]:
    """Provide mock models for testing."""
    primary = MagicMock(spec=BaseChatModel)
    primary.invoke.return_value = AIMessage(content="primary")
    primary.ainvoke.return_value = AIMessage(content="primary_async")

    fallback = MagicMock(spec=BaseChatModel)
    fallback.invoke.return_value = AIMessage(content="fallback")
    fallback.ainvoke.return_value = AIMessage(content="fallback_async")
    return primary, fallback


def test_tiered_router_routing_logic(mock_models: tuple[MagicMock, MagicMock]) -> None:
    """Test standard routing logic."""
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
async def test_tiered_router_async_logic(
    mock_models: tuple[MagicMock, MagicMock],
) -> None:
    """Test async routing logic."""
    primary, fallback = mock_models
    router = TieredSemanticRouter(primary=primary, fallback=fallback, threshold=0.5)

    await router.ainvoke("Hi")
    primary.ainvoke.assert_called_once()


def test_tiered_router_config_propagation(
    mock_models: tuple[MagicMock, MagicMock],
) -> None:
    """Ensure tags and metadata are passed down to the underlying models."""
    primary, fallback = mock_models
    router = TieredSemanticRouter(primary=primary, fallback=fallback, threshold=0.5)

    config: RunnableConfig = {"tags": ["test-tag"], "metadata": {"user_id": "123"}}
    router.invoke("Hi", config=config)

    _, kwargs = primary.invoke.call_args
    assert kwargs["config"]["tags"] == ["test-tag"]
