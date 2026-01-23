"""Tests for config isolation utilities."""

import pytest
from unittest.mock import MagicMock

from langchain_core.runnables.config import (
    var_child_runnable_config,
    isolate_config,
    aisolate_config,
    RunnableConfig,
)


class TestIsolateConfig:
    """Tests for isolate_config context manager."""

    def test_isolate_config_clears_context(self) -> None:
        """Test that isolate_config clears the parent config context."""
        # Setup: set a parent config with callbacks
        mock_callback = MagicMock()
        parent_config: RunnableConfig = {
            "callbacks": [mock_callback],
            "tags": ["parent-tag"],
            "metadata": {"source": "parent"},
        }
        var_child_runnable_config.set(parent_config)

        # Verify parent config is set
        assert var_child_runnable_config.get() == parent_config

        # Act: enter isolated context
        with isolate_config():
            # Inside isolated context, config should be None
            inner_config = var_child_runnable_config.get()
            assert inner_config is None

        # After exiting, parent config should be restored
        restored_config = var_child_runnable_config.get()
        assert restored_config == parent_config

    def test_isolate_config_restores_on_exception(self) -> None:
        """Test that isolate_config restores context even on exception."""
        parent_config: RunnableConfig = {"tags": ["test"]}
        var_child_runnable_config.set(parent_config)

        with pytest.raises(ValueError, match="test error"):
            with isolate_config():
                assert var_child_runnable_config.get() is None
                raise ValueError("test error")

        # Config should still be restored
        assert var_child_runnable_config.get() == parent_config

    def test_isolate_config_with_no_parent(self) -> None:
        """Test isolate_config when there's no parent config."""
        # Ensure no parent config
        var_child_runnable_config.set(None)

        with isolate_config():
            assert var_child_runnable_config.get() is None

        # Should remain None
        assert var_child_runnable_config.get() is None

    def test_nested_isolate_config(self) -> None:
        """Test nested isolate_config contexts."""
        outer_config: RunnableConfig = {"tags": ["outer"]}
        var_child_runnable_config.set(outer_config)

        with isolate_config():
            assert var_child_runnable_config.get() is None

            # Set a new config inside isolated context
            inner_config: RunnableConfig = {"tags": ["inner"]}
            var_child_runnable_config.set(inner_config)

            with isolate_config():
                # Nested isolation should also clear
                assert var_child_runnable_config.get() is None

            # Inner config should be restored after nested isolation
            # Note: this tests the token-based reset behavior

        # Outer config should be restored
        assert var_child_runnable_config.get() == outer_config


class TestAisolateConfig:
    """Tests for aisolate_config async context manager."""

    @pytest.mark.asyncio
    async def test_aisolate_config_clears_context(self) -> None:
        """Test that aisolate_config clears the parent config context."""
        mock_callback = MagicMock()
        parent_config: RunnableConfig = {
            "callbacks": [mock_callback],
            "tags": ["parent-tag"],
        }
        var_child_runnable_config.set(parent_config)

        async with aisolate_config():
            inner_config = var_child_runnable_config.get()
            assert inner_config is None

        restored_config = var_child_runnable_config.get()
        assert restored_config == parent_config

    @pytest.mark.asyncio
    async def test_aisolate_config_restores_on_exception(self) -> None:
        """Test that aisolate_config restores context even on exception."""
        parent_config: RunnableConfig = {"tags": ["test"]}
        var_child_runnable_config.set(parent_config)

        with pytest.raises(ValueError, match="async test error"):
            async with aisolate_config():
                assert var_child_runnable_config.get() is None
                raise ValueError("async test error")

        assert var_child_runnable_config.get() == parent_config

    @pytest.mark.asyncio
    async def test_aisolate_config_with_no_parent(self) -> None:
        """Test aisolate_config when there's no parent config."""
        var_child_runnable_config.set(None)

        async with aisolate_config():
            assert var_child_runnable_config.get() is None

        assert var_child_runnable_config.get() is None

    @pytest.mark.asyncio
    async def test_aisolate_config_with_await(self) -> None:
        """Test aisolate_config with actual async operations."""
        import asyncio

        parent_config: RunnableConfig = {"callbacks": [MagicMock()]}
        var_child_runnable_config.set(parent_config)

        async with aisolate_config():
            # Simulate async model call
            await asyncio.sleep(0.01)
            assert var_child_runnable_config.get() is None
            await asyncio.sleep(0.01)
            assert var_child_runnable_config.get() is None

        assert var_child_runnable_config.get() == parent_config
