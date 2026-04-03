"""Unit tests for ModelFallbackMiddleware."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware.model_fallback import ModelFallbackMiddleware
from langchain.agents.middleware.types import AgentState, ModelRequest, ModelResponse
from tests.unit_tests.agents.model import FakeToolCallingModel

if TYPE_CHECKING:
    from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
    from langgraph.runtime import Runtime


def _fake_runtime() -> Runtime:
    return cast("Runtime", object())


def _make_request() -> ModelRequest:
    """Create a minimal ModelRequest for testing."""
    model = GenericFakeChatModel(messages=iter([AIMessage(content="primary")]))
    return ModelRequest(
        model=model,
        system_prompt=None,
        messages=[],
        tool_choice=None,
        tools=[],
        response_format=None,
        state=AgentState(messages=[]),
        runtime=_fake_runtime(),
        model_settings={},
    )


def test_primary_model_succeeds() -> None:
    """Test that primary model is used when it succeeds."""
    primary_model = GenericFakeChatModel(messages=iter([AIMessage(content="primary response")]))
    fallback_model = GenericFakeChatModel(messages=iter([AIMessage(content="fallback response")]))

    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request()
    request = request.override(model=primary_model)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        # Simulate successful model call
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "primary response"


def test_fallback_on_primary_failure() -> None:
    """Test that fallback model is used when primary fails."""

    class FailingPrimaryModel(GenericFakeChatModel):
        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Primary model failed"
            raise ValueError(msg)

    primary_model = FailingPrimaryModel(messages=iter([AIMessage(content="should not see")]))
    fallback_model = GenericFakeChatModel(messages=iter([AIMessage(content="fallback response")]))

    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request()
    request = request.override(model=primary_model)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback response"


def test_multiple_fallbacks() -> None:
    """Test that multiple fallback models are tried in sequence."""

    class FailingModel(GenericFakeChatModel):
        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Model failed"
            raise ValueError(msg)

    primary_model = FailingModel(messages=iter([AIMessage(content="should not see")]))
    fallback1 = FailingModel(messages=iter([AIMessage(content="fallback1")]))
    fallback2 = GenericFakeChatModel(messages=iter([AIMessage(content="fallback2")]))

    middleware = ModelFallbackMiddleware(fallback1, fallback2)
    request = _make_request()
    request = request.override(model=primary_model)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback2"


def test_all_models_fail() -> None:
    """Test that exception is raised when all models fail."""

    class AlwaysFailingModel(GenericFakeChatModel):
        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Model failed"
            raise ValueError(msg)

    primary_model = AlwaysFailingModel(messages=iter([]))
    fallback_model = AlwaysFailingModel(messages=iter([]))

    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request()
    request = request.override(model=primary_model)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        result = req.model.invoke([])
        return ModelResponse(result=[result])

    with pytest.raises(ValueError, match="Model failed"):
        middleware.wrap_model_call(request, mock_handler)


async def test_primary_model_succeeds_async() -> None:
    """Test async version - primary model is used when it succeeds."""
    primary_model = GenericFakeChatModel(messages=iter([AIMessage(content="primary response")]))
    fallback_model = GenericFakeChatModel(messages=iter([AIMessage(content="fallback response")]))

    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request()
    request = request.override(model=primary_model)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        # Simulate successful async model call
        result = await req.model.ainvoke([])
        return ModelResponse(result=[result])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "primary response"


async def test_fallback_on_primary_failure_async() -> None:
    """Test async version - fallback model is used when primary fails."""

    class AsyncFailingPrimaryModel(GenericFakeChatModel):
        @override
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Primary model failed"
            raise ValueError(msg)

    primary_model = AsyncFailingPrimaryModel(messages=iter([AIMessage(content="should not see")]))
    fallback_model = GenericFakeChatModel(messages=iter([AIMessage(content="fallback response")]))

    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request()
    request = request.override(model=primary_model)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        result = await req.model.ainvoke([])
        return ModelResponse(result=[result])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback response"


async def test_multiple_fallbacks_async() -> None:
    """Test async version - multiple fallback models are tried in sequence."""

    class AsyncFailingModel(GenericFakeChatModel):
        @override
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Model failed"
            raise ValueError(msg)

    primary_model = AsyncFailingModel(messages=iter([AIMessage(content="should not see")]))
    fallback1 = AsyncFailingModel(messages=iter([AIMessage(content="fallback1")]))
    fallback2 = GenericFakeChatModel(messages=iter([AIMessage(content="fallback2")]))

    middleware = ModelFallbackMiddleware(fallback1, fallback2)
    request = _make_request()
    request = request.override(model=primary_model)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        result = await req.model.ainvoke([])
        return ModelResponse(result=[result])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback2"


async def test_all_models_fail_async() -> None:
    """Test async version - exception is raised when all models fail."""

    class AsyncAlwaysFailingModel(GenericFakeChatModel):
        @override
        async def _agenerate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: AsyncCallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Model failed"
            raise ValueError(msg)

    primary_model = AsyncAlwaysFailingModel(messages=iter([]))
    fallback_model = AsyncAlwaysFailingModel(messages=iter([]))

    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request()
    request = request.override(model=primary_model)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        result = await req.model.ainvoke([])
        return ModelResponse(result=[result])

    with pytest.raises(ValueError, match="Model failed"):
        await middleware.awrap_model_call(request, mock_handler)


def test_model_fallback_middleware_with_agent() -> None:
    """Test ModelFallbackMiddleware with agent.invoke and fallback models only."""

    class FailingModel(BaseChatModel):
        """Model that always fails."""

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = "Primary model failed"
            raise ValueError(msg)

        @property
        def _llm_type(self) -> str:
            return "failing"

    class SuccessModel(BaseChatModel):
        """Model that succeeds."""

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content="Fallback success"))]
            )

        @property
        def _llm_type(self) -> str:
            return "success"

    primary = FailingModel()
    fallback = SuccessModel()

    # Only pass fallback models to middleware (not the primary)
    fallback_middleware = ModelFallbackMiddleware(fallback)

    agent = create_agent(model=primary, middleware=[fallback_middleware])

    result = agent.invoke({"messages": [HumanMessage("Test")]})

    # Should have succeeded with fallback model
    assert len(result["messages"]) == 2
    assert result["messages"][1].content == "Fallback success"


def test_model_fallback_middleware_exhausted_with_agent() -> None:
    """Test ModelFallbackMiddleware with agent.invoke when all models fail."""

    class AlwaysFailingModel(BaseChatModel):
        """Model that always fails."""

        def __init__(self, name: str):
            super().__init__()
            self.name = name

        @override
        def _generate(
            self,
            messages: list[BaseMessage],
            stop: list[str] | None = None,
            run_manager: CallbackManagerForLLMRun | None = None,
            **kwargs: Any,
        ) -> ChatResult:
            msg = f"{self.name} failed"
            raise ValueError(msg)

        @property
        def _llm_type(self) -> str:
            return self.name or "always_failing"

    primary = AlwaysFailingModel("primary")
    fallback1 = AlwaysFailingModel("fallback1")
    fallback2 = AlwaysFailingModel("fallback2")

    # Primary fails (attempt 1), then fallback1 (attempt 2), then fallback2 (attempt 3)
    fallback_middleware = ModelFallbackMiddleware(fallback1, fallback2)

    agent = create_agent(model=primary, middleware=[fallback_middleware])

    # Should fail with the last fallback's error
    with pytest.raises(ValueError, match="fallback2 failed"):
        agent.invoke({"messages": [HumanMessage("Test")]})


def test_model_fallback_middleware_initialization() -> None:
    """Test ModelFallbackMiddleware initialization."""
    # Test with no models - now a TypeError (missing required argument)
    with pytest.raises(TypeError):
        ModelFallbackMiddleware()  # type: ignore[call-arg]

    # Test with one fallback model (valid)
    middleware = ModelFallbackMiddleware(FakeToolCallingModel())
    assert len(middleware.models) == 1

    # Test with multiple fallback models
    middleware = ModelFallbackMiddleware(FakeToolCallingModel(), FakeToolCallingModel())
    assert len(middleware.models) == 2


def test_model_request_is_frozen() -> None:
    """Test that ModelRequest raises deprecation warning on direct attribute assignment."""
    request = _make_request()
    new_model = GenericFakeChatModel(messages=iter([AIMessage(content="new model")]))

    # Direct attribute assignment should raise DeprecationWarning but still work
    with pytest.warns(
        DeprecationWarning, match="Direct attribute assignment to ModelRequest.model is deprecated"
    ):
        request.model = new_model

    # Verify the assignment actually worked
    assert request.model == new_model

    with pytest.warns(
        DeprecationWarning,
        match="Direct attribute assignment to ModelRequest.system_prompt is deprecated",
    ):
        request.system_prompt = "new prompt"  # type: ignore[misc]

    assert request.system_prompt == "new prompt"

    with pytest.warns(
        DeprecationWarning,
        match="Direct attribute assignment to ModelRequest.messages is deprecated",
    ):
        request.messages = []

    assert request.messages == []

    # Using override method should work without warnings
    request2 = _make_request()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        new_request = request2.override(
            model=new_model, system_message=SystemMessage(content="override prompt")
        )

    assert new_request.model == new_model
    assert new_request.system_prompt == "override prompt"
    # Original request should be unchanged
    assert request2.model != new_model
    assert request2.system_prompt != "override prompt"
