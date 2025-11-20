"""Tests for ConditionalModelSettingsMiddleware."""

import pytest
from langchain_core.messages import HumanMessage

from langchain.agents.middleware.conditional_model_settings import (
    ConditionalModelSettingsMiddleware,
)
from langchain.agents.middleware.types import ModelRequest
from langchain.agents.factory import create_agent

from tests.unit_tests.agents.model import FakeToolCallingModel


class TestConditionalModelSettingsBasic:
    """Test basic functionality."""

    def test_cumulative_application(self):
        """Test that multiple matching conditions apply cumulatively."""
        middleware = ConditionalModelSettingsMiddleware()

        request = ModelRequest(
            model=FakeToolCallingModel(),
            system_prompt=None,
            messages=[HumanMessage(content=f"Message {i}") for i in range(15)],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": [], "emergency": True},
            runtime=None,  # type: ignore
            model_settings={},
        )

        # First condition
        middleware._apply_settings(request, {"parallel_tool_calls": False})
        assert request.model_settings == {"parallel_tool_calls": False}

        # Second condition (cumulative)
        middleware._apply_settings(request, {"strict": True})
        assert request.model_settings == {"parallel_tool_calls": False, "strict": True}

    def test_later_settings_override_earlier(self):
        """Test that later settings override earlier ones for the same key."""
        middleware = ConditionalModelSettingsMiddleware()

        request = ModelRequest(
            model=FakeToolCallingModel(),
            system_prompt=None,
            messages=[HumanMessage(content="test")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=None,  # type: ignore
            model_settings={},
        )

        middleware._apply_settings(request, {"parallel_tool_calls": False, "strict": False})
        middleware._apply_settings(request, {"strict": True})
        assert request.model_settings == {"parallel_tool_calls": False, "strict": True}

    def test_callable_settings(self):
        """Test that callable settings are resolved correctly."""

        def compute_settings(req: ModelRequest) -> dict:
            return {"parallel_tool_calls": len(req.messages) < 5}

        request = ModelRequest(
            model=FakeToolCallingModel(),
            system_prompt=None,
            messages=[HumanMessage(content=f"Message {i}") for i in range(3)],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=None,  # type: ignore
            model_settings={},
        )

        middleware = ConditionalModelSettingsMiddleware()
        middleware._apply_settings(request, compute_settings)
        assert request.model_settings == {"parallel_tool_calls": True}

    def test_builder_pattern_chaining(self):
        """Test that builder pattern returns middleware for chaining."""
        middleware = ConditionalModelSettingsMiddleware()
        result = middleware.when(lambda req: True).use({"setting1": "value1"})
        assert result is middleware
        assert len(middleware._conditions) == 1


class TestWrapModelCall:
    """Test wrap_model_call behavior."""

    def test_multiple_conditions_cumulative(self):
        """Test that multiple conditions apply cumulatively."""
        middleware = ConditionalModelSettingsMiddleware()
        middleware.when(lambda req: len(req.messages) > 5).use({"parallel_tool_calls": False})
        middleware.when(lambda req: req.state.get("emergency")).use({"strict": True})

        request = ModelRequest(
            model=FakeToolCallingModel(),
            system_prompt=None,
            messages=[HumanMessage(content=f"Message {i}") for i in range(10)],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": [], "emergency": True},
            runtime=None,  # type: ignore
            model_settings={},
        )

        middleware.wrap_model_call(request, lambda req: None)
        assert request.model_settings == {"parallel_tool_calls": False, "strict": True}

    def test_async_condition_raises_error(self):
        """Test that async condition in sync mode raises RuntimeError."""
        middleware = ConditionalModelSettingsMiddleware()

        async def async_condition(req):
            return True

        middleware.when(async_condition).use({"parallel_tool_calls": False})

        request = ModelRequest(
            model=FakeToolCallingModel(),
            system_prompt=None,
            messages=[HumanMessage(content="test")],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=None,  # type: ignore
            model_settings={},
        )

        with pytest.raises(RuntimeError, match="Async condition function detected"):
            middleware.wrap_model_call(request, lambda req: None)


class TestAwrapModelCall:
    """Test awrap_model_call behavior."""

    @pytest.mark.asyncio
    async def test_async_condition(self):
        """Test that async conditions work."""
        middleware = ConditionalModelSettingsMiddleware()

        async def async_condition(req):
            return len(req.messages) > 5

        middleware.when(async_condition).use({"parallel_tool_calls": False})

        request = ModelRequest(
            model=FakeToolCallingModel(),
            system_prompt=None,
            messages=[HumanMessage(content=f"Message {i}") for i in range(10)],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=None,  # type: ignore
            model_settings={},
        )

        async def async_handler(req):
            return None

        await middleware.awrap_model_call(request, async_handler)
        assert request.model_settings == {"parallel_tool_calls": False}

    @pytest.mark.asyncio
    async def test_sync_condition_in_async(self):
        """Test that sync conditions work in async mode."""
        middleware = ConditionalModelSettingsMiddleware()
        middleware.when(lambda req: len(req.messages) > 5).use({"parallel_tool_calls": False})

        request = ModelRequest(
            model=FakeToolCallingModel(),
            system_prompt=None,
            messages=[HumanMessage(content=f"Message {i}") for i in range(10)],
            tool_choice=None,
            tools=[],
            response_format=None,
            state={"messages": []},
            runtime=None,  # type: ignore
            model_settings={},
        )

        async def async_handler(req):
            return None

        await middleware.awrap_model_call(request, async_handler)
        assert request.model_settings == {"parallel_tool_calls": False}


class TestCreateAgentIntegration:
    """Test middleware behavior in create_agent."""

    def test_single_condition_lambda(self):
        """Test single condition with lambda in agent."""
        captured_settings = {}

        class CapturingMiddleware(ConditionalModelSettingsMiddleware):
            def wrap_model_call(self, request, handler):
                result = super().wrap_model_call(request, handler)
                captured_settings.update(request.model_settings)
                return result

        middleware = CapturingMiddleware()
        middleware.when(lambda req: len(req.messages) > 2).use({"parallel_tool_calls": False})

        model = FakeToolCallingModel()
        agent = create_agent(model=model, middleware=[middleware])

        # Short conversation - condition should not match
        agent.invoke({"messages": [HumanMessage(content="Hello")]})
        assert "parallel_tool_calls" not in captured_settings

        # Long conversation - condition should match
        captured_settings.clear()
        agent.invoke({"messages": [HumanMessage(content=f"Msg {i}") for i in range(5)]})
        assert captured_settings.get("parallel_tool_calls") is False

    def test_single_condition_function(self):
        """Test single condition with function in agent."""
        captured_settings = {}

        class CapturingMiddleware(ConditionalModelSettingsMiddleware):
            def wrap_model_call(self, request, handler):
                result = super().wrap_model_call(request, handler)
                captured_settings.update(request.model_settings)
                return result

        def is_long_conversation(req: ModelRequest) -> bool:
            return len(req.messages) > 3

        middleware = CapturingMiddleware()
        middleware.when(is_long_conversation).use({"parallel_tool_calls": False})

        model = FakeToolCallingModel()
        agent = create_agent(model=model, middleware=[middleware])

        # Short conversation - condition should not match
        agent.invoke({"messages": [HumanMessage(content=f"Msg {i}") for i in range(2)]})
        assert "parallel_tool_calls" not in captured_settings

        # Long conversation - condition should match
        captured_settings.clear()
        agent.invoke({"messages": [HumanMessage(content=f"Msg {i}") for i in range(5)]})
        assert captured_settings.get("parallel_tool_calls") is False

    def test_multiple_conditions_cumulative(self):
        """Test multiple conditions apply cumulatively in agent."""
        captured_settings = {}

        class CapturingMiddleware(ConditionalModelSettingsMiddleware):
            def wrap_model_call(self, request, handler):
                result = super().wrap_model_call(request, handler)
                captured_settings.update(request.model_settings)
                return result

        middleware = CapturingMiddleware()
        middleware.when(lambda req: len(req.messages) > 2).use({"parallel_tool_calls": False})
        middleware.when(lambda req: len(req.messages) > 4).use({"strict": True})

        model = FakeToolCallingModel()
        agent = create_agent(model=model, middleware=[middleware])

        # Only first condition matches
        agent.invoke({"messages": [HumanMessage(content=f"Msg {i}") for i in range(3)]})
        assert captured_settings.get("parallel_tool_calls") is False
        assert "strict" not in captured_settings

        # Both conditions match - should apply both settings
        captured_settings.clear()
        agent.invoke({"messages": [HumanMessage(content=f"Msg {i}") for i in range(6)]})
        assert captured_settings.get("parallel_tool_calls") is False
        assert captured_settings.get("strict") is True

    def test_callable_settings_function(self):
        """Test callable settings (use with function) in agent."""
        captured_settings = {}

        class CapturingMiddleware(ConditionalModelSettingsMiddleware):
            def wrap_model_call(self, request, handler):
                result = super().wrap_model_call(request, handler)
                captured_settings.update(request.model_settings)
                return result

        def compute_settings(req: ModelRequest) -> dict:
            return {"parallel_tool_calls": len(req.messages) < 3}

        middleware = CapturingMiddleware()
        middleware.when(lambda req: True).use(compute_settings)

        model = FakeToolCallingModel()
        agent = create_agent(model=model, middleware=[middleware])

        # Short conversation - should enable parallel
        agent.invoke({"messages": [HumanMessage(content="Test")]})
        assert captured_settings.get("parallel_tool_calls") is True

        # Long conversation - should disable parallel
        captured_settings.clear()
        agent.invoke({"messages": [HumanMessage(content=f"Msg {i}") for i in range(5)]})
        assert captured_settings.get("parallel_tool_calls") is False
