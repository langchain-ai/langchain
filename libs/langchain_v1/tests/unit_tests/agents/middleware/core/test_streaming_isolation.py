"""Tests for middleware streaming callback isolation.

When an agent is executed using streaming (astream), model calls made inside
middleware hooks should NOT inherit the parent agent's streaming callbacks.
This prevents tokens from middleware-internal model calls from leaking into
the parent stream.

See: https://github.com/langchain-ai/langchain/issues/34382
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables.config import (
    RunnableConfig,
    ensure_config,
    var_child_runnable_config,
)
from langgraph.runtime import Runtime
from typing_extensions import override

from langchain.agents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import AgentMiddleware, AgentState


class CallbackCapturingModel(BaseChatModel):
    """A model that captures the config it receives."""

    captured_configs: list[RunnableConfig | None] = []  # noqa: RUF012

    @override
    def invoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.captured_configs.append(config)
        return AIMessage(content="Test summary response")

    @override
    async def ainvoke(
        self,
        input: Any,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.captured_configs.append(config)
        return AIMessage(content="Test summary response")

    @override
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="Test summary"))])

    @property
    def _llm_type(self) -> str:
        return "callback-capturing"


class TestGetInternalModelConfig:
    """Tests for AgentMiddleware._get_internal_model_config."""

    def test_returns_empty_callbacks(self) -> None:
        """Config should have callbacks=[] to prevent inheritance."""
        config = AgentMiddleware._get_internal_model_config()
        assert config["callbacks"] == []

    def test_includes_metadata(self) -> None:
        """Metadata kwargs should be included in the config."""
        config = AgentMiddleware._get_internal_model_config(
            lc_source="summarization", custom_key="value"
        )
        assert config["callbacks"] == []
        assert config["metadata"] == {
            "lc_source": "summarization",
            "custom_key": "value",
        }

    def test_no_metadata_when_not_provided(self) -> None:
        """Config should not include metadata key when no kwargs are passed."""
        config = AgentMiddleware._get_internal_model_config()
        assert "metadata" not in config

    def test_overrides_inherited_streaming_callbacks(self) -> None:
        """Config with callbacks=[] should override var_child_runnable_config callbacks.

        When ensure_config merges configs, explicit callbacks=[] should take
        precedence over inherited streaming callbacks from the parent context.
        """
        # Simulate parent context with streaming callbacks
        fake_callback = MagicMock()
        parent_config: RunnableConfig = {"callbacks": [fake_callback]}
        token = var_child_runnable_config.set(parent_config)
        try:
            # Without isolation: ensure_config inherits the streaming callback
            inherited = ensure_config(None)
            assert inherited["callbacks"] == [fake_callback]

            # With isolation: our config overrides the callbacks
            internal_config = AgentMiddleware._get_internal_model_config(lc_source="test")
            isolated = ensure_config(internal_config)
            assert isolated["callbacks"] == []
        finally:
            var_child_runnable_config.reset(token)


class TestSummarizationMiddlewareStreamingIsolation:
    """Tests that SummarizationMiddleware isolates internal model calls."""

    def test_create_summary_uses_isolated_config(self) -> None:
        """_create_summary should pass config with callbacks=[] to model.invoke."""
        model = CallbackCapturingModel()
        middleware = SummarizationMiddleware(
            model=model,
            trigger=("tokens", 100),
        )

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        middleware._create_summary(messages)

        assert len(model.captured_configs) == 1
        config = model.captured_configs[0]
        assert config is not None
        assert config["callbacks"] == []
        assert config["metadata"] == {"lc_source": "summarization"}

    @pytest.mark.asyncio
    async def test_acreate_summary_uses_isolated_config(self) -> None:
        """_acreate_summary should pass config with callbacks=[] to model.ainvoke."""
        model = CallbackCapturingModel()
        middleware = SummarizationMiddleware(
            model=model,
            trigger=("tokens", 100),
        )

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]
        await middleware._acreate_summary(messages)

        assert len(model.captured_configs) == 1
        config = model.captured_configs[0]
        assert config is not None
        assert config["callbacks"] == []
        assert config["metadata"] == {"lc_source": "summarization"}

    def test_create_summary_does_not_inherit_parent_callbacks(self) -> None:
        """Verify _create_summary does not inherit parent streaming callbacks.

        When var_child_runnable_config has streaming callbacks,
        _create_summary should NOT pass those to the model.
        """
        model = CallbackCapturingModel()
        middleware = SummarizationMiddleware(
            model=model,
            trigger=("tokens", 100),
        )

        messages = [
            HumanMessage(content="Hello"),
            AIMessage(content="Hi there!"),
        ]

        # Simulate parent context with streaming callbacks
        fake_streaming_callback = MagicMock()
        parent_config: RunnableConfig = {"callbacks": [fake_streaming_callback]}
        token = var_child_runnable_config.set(parent_config)
        try:
            middleware._create_summary(messages)
        finally:
            var_child_runnable_config.reset(token)

        assert len(model.captured_configs) == 1
        config = model.captured_configs[0]
        assert config is not None
        # The config should have empty callbacks, NOT the parent's streaming callback
        assert config["callbacks"] == []


class TestCustomMiddlewareStreamingIsolation:
    """Tests that custom middleware can use _get_internal_model_config."""

    def test_custom_middleware_can_isolate_model_calls(self) -> None:
        """Custom middleware subclass should be able to use _get_internal_model_config."""
        model = CallbackCapturingModel()

        class CustomMiddleware(AgentMiddleware):
            def __init__(self) -> None:
                self.model = model

            def before_model(self, state: AgentState[Any], runtime: Runtime) -> None:
                self.model.invoke(
                    "internal prompt",
                    config=self._get_internal_model_config(lc_source="custom_middleware"),
                )

        middleware = CustomMiddleware()

        # Simulate parent context with streaming callbacks
        fake_callback = MagicMock()
        parent_config: RunnableConfig = {"callbacks": [fake_callback]}
        token = var_child_runnable_config.set(parent_config)
        try:
            state = AgentState[Any](messages=[HumanMessage(content="Hello")])
            middleware.before_model(state, Runtime())
        finally:
            var_child_runnable_config.reset(token)

        assert len(model.captured_configs) == 1
        config = model.captured_configs[0]
        assert config is not None
        assert config["callbacks"] == []
        assert config["metadata"] == {"lc_source": "custom_middleware"}
