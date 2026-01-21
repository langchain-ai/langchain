"""Tests for callback isolation in middleware internal model calls.

These tests verify that the get_internal_call_config() utility correctly
filters callbacks to:
- PRESERVE tracing callbacks (LangSmith, custom tracers)
- PRESERVE usage tracking callbacks
- BLOCK streaming callbacks (to prevent token leakage)
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch
from uuid import UUID

import pytest
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_core.tracers.base import BaseTracer
from langchain_core.tracers.schemas import Run

from langchain.agents.middleware._callback_utils import get_internal_call_config


class MockTracer(BaseTracer):
    """Mock tracer for testing - simulates LangSmith tracer."""

    def __init__(self) -> None:
        super().__init__()
        self.runs: list[Run] = []

    def _persist_run(self, run: Run) -> None:
        self.runs.append(run)


class MockStreamingHandler(StreamingStdOutCallbackHandler):
    """Mock streaming handler for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.tokens: list[str] = []

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        self.tokens.append(token)


class MockUsageHandler(BaseCallbackHandler):
    """Mock usage tracking handler for testing."""

    def __init__(self) -> None:
        super().__init__()
        self.total_tokens = 0

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        self.total_tokens += 1


class CustomObservabilityHandler(BaseCallbackHandler):
    """Custom handler that user wants to propagate to internal calls."""

    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def on_llm_start(self, *args: Any, **kwargs: Any) -> None:
        self.events.append("llm_start")


class TestGetInternalCallConfig:
    """Tests for get_internal_call_config utility."""

    def test_no_callbacks_returns_empty_list(self) -> None:
        """When no callbacks in context, return empty list."""
        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": None},
        ):
            config = get_internal_call_config()
            assert config == {"callbacks": []}

    def test_empty_callbacks_returns_empty_list(self) -> None:
        """When empty callbacks list in context, return empty list."""
        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": []},
        ):
            config = get_internal_call_config()
            assert config == {"callbacks": []}

    def test_tracer_is_preserved(self) -> None:
        """BaseTracer subclasses should be preserved for LangSmith observability."""
        tracer = MockTracer()
        streaming_handler = MockStreamingHandler()

        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": [tracer, streaming_handler]},
        ):
            config = get_internal_call_config()
            callbacks = config["callbacks"]

            assert tracer in callbacks, "Tracer should be preserved"
            assert streaming_handler not in callbacks, "Streaming handler should be blocked"

    def test_streaming_handler_is_blocked(self) -> None:
        """StreamingStdOutCallbackHandler and similar should be blocked."""
        streaming_handler = MockStreamingHandler()

        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": [streaming_handler]},
        ):
            config = get_internal_call_config()
            callbacks = config["callbacks"]

            assert len(callbacks) == 0, "Streaming handler should be blocked"

    def test_usage_handler_is_preserved(self) -> None:
        """UsageMetadataCallbackHandler should be preserved for cost tracking."""
        # Mock the UsageMetadataCallbackHandler import
        mock_usage_class = type(
            "UsageMetadataCallbackHandler",
            (BaseCallbackHandler,),
            {},
        )
        mock_usage_handler = mock_usage_class()

        with patch(
            "langchain.agents.middleware._callback_utils._get_usage_handler_type",
            return_value=mock_usage_class,
        ):
            with patch(
                "langchain.agents.middleware._callback_utils.ensure_config",
                return_value={"callbacks": [mock_usage_handler]},
            ):
                config = get_internal_call_config()
                callbacks = config["callbacks"]

                assert mock_usage_handler in callbacks, "Usage handler should be preserved"

    def test_additional_callback_types_preserved(self) -> None:
        """User-specified additional callback types should be preserved."""
        custom_handler = CustomObservabilityHandler()
        streaming_handler = MockStreamingHandler()

        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": [custom_handler, streaming_handler]},
        ):
            config = get_internal_call_config(
                additional_callback_types=[CustomObservabilityHandler]
            )
            callbacks = config["callbacks"]

            assert custom_handler in callbacks, "Custom handler should be preserved"
            assert streaming_handler not in callbacks, "Streaming handler should be blocked"

    def test_mixed_callbacks_filtered_correctly(self) -> None:
        """Test realistic scenario with multiple callback types."""
        tracer = MockTracer()
        streaming_handler = MockStreamingHandler()
        generic_handler = BaseCallbackHandler()

        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": [tracer, streaming_handler, generic_handler]},
        ):
            config = get_internal_call_config()
            callbacks = config["callbacks"]

            assert tracer in callbacks, "Tracer should be preserved"
            assert streaming_handler not in callbacks, "Streaming should be blocked"
            assert generic_handler not in callbacks, "Generic handler should be blocked"

    def test_callback_manager_handlers_extracted(self) -> None:
        """Test that handlers are correctly extracted from CallbackManager."""
        tracer = MockTracer()
        streaming_handler = MockStreamingHandler()

        # Mock a CallbackManager-like object
        mock_manager = MagicMock()
        mock_manager.handlers = [tracer, streaming_handler]

        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": mock_manager},
        ):
            config = get_internal_call_config()
            callbacks = config["callbacks"]

            assert tracer in callbacks, "Tracer should be preserved"
            assert streaming_handler not in callbacks, "Streaming should be blocked"

    def test_unknown_callback_type_returns_empty(self) -> None:
        """Unknown callback container types should return empty to be safe."""
        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": "invalid_type"},
        ):
            config = get_internal_call_config()
            assert config == {"callbacks": []}


class TestCallbackIsolationIntegration:
    """Integration tests for callback isolation in actual middleware usage."""

    def test_tracer_sees_internal_calls_streaming_does_not(self) -> None:
        """Verify the core contract: tracers see internal calls, streaming doesn't."""
        tracer = MockTracer()
        streaming_handler = MockStreamingHandler()

        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": [tracer, streaming_handler]},
        ):
            config = get_internal_call_config()

            # Verify tracer is included (would see LangSmith events)
            assert any(
                isinstance(cb, BaseTracer) for cb in config["callbacks"]
            ), "At least one tracer should be preserved"

            # Verify no streaming handlers (would leak tokens)
            assert not any(
                isinstance(cb, StreamingStdOutCallbackHandler) for cb in config["callbacks"]
            ), "No streaming handlers should be present"

    def test_multiple_tracers_all_preserved(self) -> None:
        """Multiple tracer instances should all be preserved."""
        tracer1 = MockTracer()
        tracer2 = MockTracer()

        with patch(
            "langchain.agents.middleware._callback_utils.ensure_config",
            return_value={"callbacks": [tracer1, tracer2]},
        ):
            config = get_internal_call_config()
            callbacks = config["callbacks"]

            assert tracer1 in callbacks
            assert tracer2 in callbacks
            assert len(callbacks) == 2
