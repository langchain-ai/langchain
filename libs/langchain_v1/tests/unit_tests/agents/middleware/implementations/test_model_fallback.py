"""Unit tests for ModelFallbackMiddleware."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, tool
from typing_extensions import override

from langchain.agents.factory import create_agent
from langchain.agents.middleware import model_fallback as model_fallback_module
from langchain.agents.middleware.model_fallback import (
    ModelFallbackMiddleware,
    _sanitize_request_for_fallback,
    _supports_anthropic_cache_control,
)
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


def _make_request_with_cache_markers(primary_model: BaseChatModel) -> ModelRequest:
    """Create a request with Anthropic-style cache markers in all relevant fields."""

    @tool(extras={"cache_control": {"type": "ephemeral"}, "defer_loading": True})
    def cached_tool(query: str) -> str:
        """Tool used for cache marker sanitization tests."""
        return query

    return _make_request().override(
        model=primary_model,
        model_settings={
            "temperature": 0.3,
            "top_p": 0.8,
            "cache_control": {"type": "ephemeral"},
        },
        system_message=SystemMessage(
            content=[
                {"type": "text", "text": "policy", "cache_control": {"type": "ephemeral"}},
                {
                    "type": "text",
                    "text": "extra",
                    "extras": {
                        "priority": "high",
                        "cache_control": {"type": "ephemeral"},
                    },
                },
            ]
        ),
        messages=[
            HumanMessage(
                content=[
                    {
                        "type": "text",
                        "text": "question",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": "details",
                        "metadata": {
                            "source": "user",
                            "cache_control": {"type": "ephemeral"},
                        },
                    },
                ]
            ),
            AIMessage(
                content=[
                    {
                        "type": "text",
                        "text": "previous",
                        "extras": {
                            "turn": 1,
                            "cache_control": {"type": "ephemeral"},
                        },
                        "metadata": {
                            "source": "assistant",
                            "cache_control": {"type": "ephemeral"},
                        },
                    },
                ]
            ),
        ],
        tools=[
            cached_tool,
            {
                "name": "dict_tool",
                "description": "dict-style tool payload",
                "input_schema": {"type": "object", "properties": {}},
                "cache_control": {"type": "ephemeral"},
                "extras": {
                    "defer_loading": True,
                    "cache_control": {"type": "ephemeral"},
                },
            },
        ],
    )


def _assert_request_has_cache_markers(request: ModelRequest) -> None:
    """Assert request still contains Anthropic-style cache markers."""
    assert "cache_control" in request.model_settings
    assert request.system_message is not None
    system_content = request.system_message.content
    assert isinstance(system_content, list)
    assert isinstance(system_content[0], dict)
    assert "cache_control" in system_content[0]
    assert isinstance(system_content[1], dict)
    system_extras = system_content[1].get("extras")
    assert isinstance(system_extras, dict)
    assert "cache_control" in system_extras

    first_message_content = request.messages[0].content
    assert isinstance(first_message_content, list)
    assert isinstance(first_message_content[0], dict)
    assert "cache_control" in first_message_content[0]
    assert isinstance(first_message_content[1], dict)
    first_message_metadata = first_message_content[1].get("metadata")
    assert isinstance(first_message_metadata, dict)
    assert "cache_control" in first_message_metadata

    second_message_content = request.messages[1].content
    assert isinstance(second_message_content, list)
    assert isinstance(second_message_content[0], dict)
    second_message_extras = second_message_content[0].get("extras")
    assert isinstance(second_message_extras, dict)
    assert "cache_control" in second_message_extras
    second_message_metadata = second_message_content[0].get("metadata")
    assert isinstance(second_message_metadata, dict)
    assert "cache_control" in second_message_metadata

    base_tool = request.tools[0]
    assert isinstance(base_tool, BaseTool)
    assert base_tool.extras is not None
    assert "cache_control" in base_tool.extras

    dict_tool = request.tools[1]
    assert isinstance(dict_tool, dict)
    assert "cache_control" in dict_tool
    dict_tool_extras = dict_tool.get("extras")
    assert isinstance(dict_tool_extras, dict)
    assert "cache_control" in dict_tool_extras


def _assert_request_is_sanitized(request: ModelRequest) -> None:
    """Assert request has cache markers removed while preserving unrelated data."""
    assert request.model_settings == {"temperature": 0.3, "top_p": 0.8}

    assert request.system_message is not None
    system_content = request.system_message.content
    assert isinstance(system_content, list)
    assert all(
        not (isinstance(block, dict) and "cache_control" in block) for block in system_content
    )
    assert isinstance(system_content[1], dict)
    assert system_content[1].get("extras") == {"priority": "high"}

    for message in request.messages:
        message_content = message.content
        if isinstance(message_content, list):
            assert all(
                not (isinstance(block, dict) and "cache_control" in block)
                for block in message_content
            )

    first_message_content = request.messages[0].content
    assert isinstance(first_message_content, list)
    assert isinstance(first_message_content[1], dict)
    assert first_message_content[1].get("metadata") == {"source": "user"}

    second_message_content = request.messages[1].content
    assert isinstance(second_message_content, list)
    assert isinstance(second_message_content[0], dict)
    assert second_message_content[0].get("extras") == {"turn": 1}
    assert second_message_content[0].get("metadata") == {"source": "assistant"}

    base_tool = request.tools[0]
    assert isinstance(base_tool, BaseTool)
    assert base_tool.extras is not None
    assert base_tool.extras == {"defer_loading": True}

    dict_tool = request.tools[1]
    assert isinstance(dict_tool, dict)
    assert "cache_control" not in dict_tool
    dict_tool_extras = dict_tool.get("extras")
    assert dict_tool_extras == {"defer_loading": True}


def test_fallback_sanitizes_cache_markers_sync() -> None:
    """Fallback attempts should strip Anthropic cache markers in sync path."""
    primary_model = GenericFakeChatModel(messages=iter([AIMessage(content="primary response")]))
    fallback_model = GenericFakeChatModel(messages=iter([AIMessage(content="fallback response")]))
    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)

        assert req.model is fallback_model
        _assert_request_is_sanitized(req)
        return ModelResponse(result=[AIMessage(content="fallback response")])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback response"
    assert len(attempts) == 2
    _assert_request_has_cache_markers(request)


async def test_fallback_sanitizes_cache_markers_async() -> None:
    """Fallback attempts should strip Anthropic cache markers in async path."""
    primary_model = GenericFakeChatModel(messages=iter([AIMessage(content="primary response")]))
    fallback_model = GenericFakeChatModel(messages=iter([AIMessage(content="fallback response")]))
    middleware = ModelFallbackMiddleware(fallback_model)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)

        assert req.model is fallback_model
        _assert_request_is_sanitized(req)
        return ModelResponse(result=[AIMessage(content="fallback response")])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback response"
    assert len(attempts) == 2
    _assert_request_has_cache_markers(request)


def test_sanitize_collapses_emptied_extras_to_none() -> None:
    """Stripping the only `extras` key (`cache_control`) resets `extras` to None."""

    @tool(extras={"cache_control": {"type": "ephemeral"}})
    def base_tool_only_cache(query: str) -> str:
        """Tool whose extras hold only a cache marker."""
        return query

    request = _make_request().override(
        tools=[
            base_tool_only_cache,
            {
                "name": "dict_tool",
                "description": "dict-style tool payload",
                "extras": {"cache_control": {"type": "ephemeral"}},
            },
        ],
    )

    sanitized = _sanitize_request_for_fallback(request)

    base_tool = sanitized.tools[0]
    assert isinstance(base_tool, BaseTool)
    assert base_tool.extras is None

    dict_tool = sanitized.tools[1]
    assert isinstance(dict_tool, dict)
    assert dict_tool.get("extras") is None


def test_sanitize_returns_same_request_when_no_markers() -> None:
    """A marker-free request passes through as the same instance (no copies)."""

    @tool
    def plain_tool(query: str) -> str:
        """Marker-free tool."""
        return query

    request = _make_request().override(
        model_settings={"temperature": 0.5},
        system_message=SystemMessage(content=[{"type": "text", "text": "policy"}]),
        messages=[HumanMessage(content=[{"type": "text", "text": "question"}])],
        tools=[plain_tool],
    )

    assert _sanitize_request_for_fallback(request) is request


def test_sanitize_preserves_identity_of_unchanged_fields() -> None:
    """Only fields containing markers are copied; the rest keep object identity."""
    request = _make_request().override(
        model_settings={"temperature": 0.5, "cache_control": {"type": "ephemeral"}},
        system_message=SystemMessage(content=[{"type": "text", "text": "policy"}]),
        messages=[HumanMessage(content=[{"type": "text", "text": "question"}])],
    )

    sanitized = _sanitize_request_for_fallback(request)

    # Only `model_settings` contained a marker, so everything else is untouched.
    assert sanitized is not request
    assert sanitized.model_settings == {"temperature": 0.5}
    assert sanitized.system_message is request.system_message
    assert sanitized.messages is request.messages
    assert sanitized.tools is request.tools


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


class _FakeAnthropicModel(GenericFakeChatModel):
    """Fake model that reports `anthropic-chat` as its `_llm_type`.

    This simulates a direct-Anthropic model for provider-based cache-control
    detection without importing the real `ChatAnthropic` (which requires
    `langchain-anthropic`).
    """

    @property
    def _llm_type(self) -> str:
        return "anthropic-chat"


class _FakeBedrockAnthropicModel(GenericFakeChatModel):
    """Fake model that reports `anthropic-bedrock-chat` as its `_llm_type`.

    Simulates a Bedrock-hosted Claude model. `ChatAnthropic._get_request_payload`
    translates the top-level `cache_control` kwarg into a block-level breakpoint
    for this `_llm_type`, while content-block and tool markers pass through, so
    cache markers are valid for this provider.
    """

    @property
    def _llm_type(self) -> str:
        return "anthropic-bedrock-chat"


class _FakeVertexAnthropicModel(GenericFakeChatModel):
    """Fake model that reports `anthropic-chat-vertexai` as its `_llm_type`."""

    @property
    def _llm_type(self) -> str:
        return "anthropic-chat-vertexai"


class _FakeNonStringLlmTypeModel(GenericFakeChatModel):
    """Fake whose `_llm_type` is not a string, exercising the `isinstance` guard.

    A list is deliberately unhashable, so without the guard the frozenset
    membership test would raise `TypeError` rather than return `False`.
    """

    @property
    def _llm_type(self) -> str:
        return ["anthropic-chat"]  # type: ignore[return-value]


_ANTHROPIC_COMPATIBLE_FAKES = [
    _FakeAnthropicModel,
    _FakeBedrockAnthropicModel,
    _FakeVertexAnthropicModel,
]


def test_supports_anthropic_cache_control() -> None:
    """`_supports_anthropic_cache_control` detects Anthropic-compatible models."""
    assert _supports_anthropic_cache_control(_FakeAnthropicModel(messages=iter([])))
    assert _supports_anthropic_cache_control(_FakeBedrockAnthropicModel(messages=iter([])))
    assert _supports_anthropic_cache_control(_FakeVertexAnthropicModel(messages=iter([])))
    assert not _supports_anthropic_cache_control(GenericFakeChatModel(messages=iter([])))
    assert not _supports_anthropic_cache_control(FakeToolCallingModel())
    # A non-string `_llm_type` must be rejected by the guard rather than raising.
    assert not _supports_anthropic_cache_control(_FakeNonStringLlmTypeModel(messages=iter([])))


def test_fallback_preserves_cache_markers_for_anthropic_sync() -> None:
    """Anthropic fallback keeps cache markers; non-Anthropic fallback strips them."""
    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    anthropic_fallback = _FakeAnthropicModel(
        messages=iter([AIMessage(content="anthropic fallback")])
    )
    non_anthropic_fallback = GenericFakeChatModel(
        messages=iter([AIMessage(content="non-anthropic fallback")])
    )
    middleware = ModelFallbackMiddleware(anthropic_fallback, non_anthropic_fallback)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            # Primary attempt — markers present
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)
        if len(attempts) == 2:
            # Anthropic fallback — markers preserved
            assert req.model is anthropic_fallback
            _assert_request_has_cache_markers(req)
            msg = "Anthropic fallback failed"
            raise ValueError(msg)
        # Non-Anthropic fallback — markers stripped
        assert req.model is non_anthropic_fallback
        _assert_request_is_sanitized(req)
        return ModelResponse(result=[AIMessage(content="non-anthropic fallback")])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "non-anthropic fallback"
    assert len(attempts) == 3
    _assert_request_has_cache_markers(request)


async def test_fallback_preserves_cache_markers_for_anthropic_async() -> None:
    """Async: Anthropic fallback keeps cache markers; non-Anthropic strips them."""
    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    anthropic_fallback = _FakeAnthropicModel(
        messages=iter([AIMessage(content="anthropic fallback")])
    )
    non_anthropic_fallback = GenericFakeChatModel(
        messages=iter([AIMessage(content="non-anthropic fallback")])
    )
    middleware = ModelFallbackMiddleware(anthropic_fallback, non_anthropic_fallback)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)
        if len(attempts) == 2:
            assert req.model is anthropic_fallback
            _assert_request_has_cache_markers(req)
            msg = "Anthropic fallback failed"
            raise ValueError(msg)
        assert req.model is non_anthropic_fallback
        _assert_request_is_sanitized(req)
        return ModelResponse(result=[AIMessage(content="non-anthropic fallback")])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "non-anthropic fallback"
    assert len(attempts) == 3
    _assert_request_has_cache_markers(request)


@pytest.mark.parametrize("fallback_cls", _ANTHROPIC_COMPATIBLE_FAKES)
def test_fallback_preserves_cache_markers_for_anthropic_compatible_sync(
    fallback_cls: type[GenericFakeChatModel],
) -> None:
    """Any Anthropic-compatible fallback (direct, Bedrock, Vertex) keeps markers."""
    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    fallback = fallback_cls(messages=iter([AIMessage(content="fallback")]))
    middleware = ModelFallbackMiddleware(fallback)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)
        # Anthropic-compatible fallback — markers preserved
        assert req.model is fallback
        _assert_request_has_cache_markers(req)
        return ModelResponse(result=[AIMessage(content="fallback")])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback"
    assert len(attempts) == 2
    _assert_request_has_cache_markers(request)


@pytest.mark.parametrize("fallback_cls", _ANTHROPIC_COMPATIBLE_FAKES)
async def test_fallback_preserves_cache_markers_for_anthropic_compatible_async(
    fallback_cls: type[GenericFakeChatModel],
) -> None:
    """Async: any Anthropic-compatible fallback (direct, Bedrock, Vertex) keeps markers."""
    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    fallback = fallback_cls(messages=iter([AIMessage(content="fallback")]))
    middleware = ModelFallbackMiddleware(fallback)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)
        assert req.model is fallback
        _assert_request_has_cache_markers(req)
        return ModelResponse(result=[AIMessage(content="fallback")])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "fallback"
    assert len(attempts) == 2
    _assert_request_has_cache_markers(request)


def test_fallback_reverse_order_preserves_anthropic_markers_sync() -> None:
    """A non-Anthropic fallback first must not corrupt a later Anthropic fallback.

    Each iteration derives its request from the original, so the Anthropic
    fallback still sees cache markers even though the earlier non-Anthropic
    fallback received a sanitized request. Guards against a regression to
    loop-carried reassignment of the request.
    """
    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    non_anthropic_fallback = GenericFakeChatModel(
        messages=iter([AIMessage(content="non-anthropic fallback")])
    )
    anthropic_fallback = _FakeAnthropicModel(
        messages=iter([AIMessage(content="anthropic fallback")])
    )
    middleware = ModelFallbackMiddleware(non_anthropic_fallback, anthropic_fallback)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)
        if len(attempts) == 2:
            # Non-Anthropic fallback first — markers stripped
            assert req.model is non_anthropic_fallback
            _assert_request_is_sanitized(req)
            msg = "Non-Anthropic fallback failed"
            raise ValueError(msg)
        # Anthropic fallback second — markers still present (derived from original)
        assert req.model is anthropic_fallback
        _assert_request_has_cache_markers(req)
        return ModelResponse(result=[AIMessage(content="anthropic fallback")])

    response = middleware.wrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "anthropic fallback"
    assert len(attempts) == 3
    _assert_request_has_cache_markers(request)


async def test_fallback_reverse_order_preserves_anthropic_markers_async() -> None:
    """Async: non-Anthropic fallback first must not corrupt a later Anthropic fallback."""
    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    non_anthropic_fallback = GenericFakeChatModel(
        messages=iter([AIMessage(content="non-anthropic fallback")])
    )
    anthropic_fallback = _FakeAnthropicModel(
        messages=iter([AIMessage(content="anthropic fallback")])
    )
    middleware = ModelFallbackMiddleware(non_anthropic_fallback, anthropic_fallback)
    request = _make_request_with_cache_markers(primary_model)
    attempts: list[ModelRequest] = []

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            _assert_request_has_cache_markers(req)
            msg = "Primary model failed"
            raise ValueError(msg)
        if len(attempts) == 2:
            assert req.model is non_anthropic_fallback
            _assert_request_is_sanitized(req)
            msg = "Non-Anthropic fallback failed"
            raise ValueError(msg)
        assert req.model is anthropic_fallback
        _assert_request_has_cache_markers(req)
        return ModelResponse(result=[AIMessage(content="anthropic fallback")])

    response = await middleware.awrap_model_call(request, mock_handler)

    assert isinstance(response, ModelResponse)
    assert response.result[0].content == "anthropic fallback"
    assert len(attempts) == 3
    _assert_request_has_cache_markers(request)


def test_fallback_sanitizer_error_is_not_masked_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A sanitizer bug must surface, not be swallowed and hidden by a later success.

    The sanitized request is built outside the ``try`` that guards the model
    call, so an exception from sanitization propagates immediately instead of
    being caught, recorded as a model failure, and masked when a subsequent
    Anthropic fallback succeeds.
    """

    def _boom(_request: ModelRequest) -> ModelRequest:
        msg = "sanitizer boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(model_fallback_module, "_sanitize_request_for_fallback", _boom)

    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    non_anthropic_fallback = GenericFakeChatModel(
        messages=iter([AIMessage(content="non-anthropic fallback")])
    )
    anthropic_fallback = _FakeAnthropicModel(
        messages=iter([AIMessage(content="anthropic fallback")])
    )
    middleware = ModelFallbackMiddleware(non_anthropic_fallback, anthropic_fallback)
    request = _make_request_with_cache_markers(primary_model)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        if req.model is primary_model:
            msg = "Primary model failed"
            raise ValueError(msg)
        # The Anthropic fallback must never be reached: the sanitizer error on
        # the preceding non-Anthropic fallback should have propagated first.
        return ModelResponse(result=[AIMessage(content="should not be reached")])

    with pytest.raises(RuntimeError, match="sanitizer boom"):
        middleware.wrap_model_call(request, mock_handler)


async def test_fallback_sanitizer_error_is_not_masked_async(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Async: a sanitizer bug must surface, not be masked by a later success."""

    def _boom(_request: ModelRequest) -> ModelRequest:
        msg = "sanitizer boom"
        raise RuntimeError(msg)

    monkeypatch.setattr(model_fallback_module, "_sanitize_request_for_fallback", _boom)

    primary_model = _FakeAnthropicModel(messages=iter([AIMessage(content="primary response")]))
    non_anthropic_fallback = GenericFakeChatModel(
        messages=iter([AIMessage(content="non-anthropic fallback")])
    )
    anthropic_fallback = _FakeAnthropicModel(
        messages=iter([AIMessage(content="anthropic fallback")])
    )
    middleware = ModelFallbackMiddleware(non_anthropic_fallback, anthropic_fallback)
    request = _make_request_with_cache_markers(primary_model)

    async def mock_handler(req: ModelRequest) -> ModelResponse:
        if req.model is primary_model:
            msg = "Primary model failed"
            raise ValueError(msg)
        return ModelResponse(result=[AIMessage(content="should not be reached")])

    with pytest.raises(RuntimeError, match="sanitizer boom"):
        await middleware.awrap_model_call(request, mock_handler)


def test_lazy_initialization_does_not_instantiate_on_init(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No model should be instantiated during middleware construction."""
    called = False

    def fake_init(_spec: str) -> BaseChatModel:
        nonlocal called
        called = True
        return GenericFakeChatModel(messages=iter([AIMessage(content="lazy")]))

    monkeypatch.setattr(model_fallback_module, "init_chat_model", fake_init)

    # Constructing the middleware must not call `init_chat_model`.
    ModelFallbackMiddleware("fake:model")
    assert not called


def test_lazy_initialization_and_caching(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """First access should initialize once; subsequent uses reuse cached model."""
    call_count = 0

    def fake_init(_spec: str) -> BaseChatModel:
        nonlocal call_count
        call_count += 1
        return GenericFakeChatModel(messages=iter([AIMessage(content=f"inst{call_count}")]))

    monkeypatch.setattr(model_fallback_module, "init_chat_model", fake_init)

    middleware = ModelFallbackMiddleware("fake:model")

    primary_model = GenericFakeChatModel(messages=iter([AIMessage(content="primary")]))
    request = _make_request().override(model=primary_model)

    attempts: list[ModelRequest] = []

    def mock_handler(req: ModelRequest) -> ModelResponse:
        attempts.append(req)
        if len(attempts) == 1:
            msg = "Primary failed"
            raise ValueError(msg)
        # The fallback should be an instantiated BaseChatModel
        assert isinstance(req.model, BaseChatModel)
        return ModelResponse(result=[AIMessage(content="fallback")])

    response = middleware.wrap_model_call(request, mock_handler)
    assert isinstance(response, ModelResponse)
    assert call_count == 1

    # Second overall call should reuse cached instance (no extra init)
    attempts.clear()
    response = middleware.wrap_model_call(request, mock_handler)
    assert call_count == 1


def test_passing_basechatmodel_is_reused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing a BaseChatModel instance should bypass initialization entirely."""
    call_count = 0

    def fake_init(_spec: str) -> BaseChatModel:
        nonlocal call_count
        call_count += 1
        return GenericFakeChatModel(messages=iter([AIMessage(content=f"inst{call_count}")]))

    monkeypatch.setattr(model_fallback_module, "init_chat_model", fake_init)

    instance = GenericFakeChatModel(messages=iter([AIMessage(content="fallback instance")]))
    middleware = ModelFallbackMiddleware(instance)

    # init_chat_model must not have been called
    assert call_count == 0

    primary_model = GenericFakeChatModel(messages=iter([AIMessage(content="primary")]))
    request = _make_request().override(model=primary_model)

    def mock_handler(req: ModelRequest) -> ModelResponse:
        if req.model is primary_model:
            msg = "Primary failed"
            raise ValueError(msg)
        # Ensure the exact instance passed to middleware is used
        assert req.model is instance
        return ModelResponse(result=[AIMessage(content="ok")])

    response = middleware.wrap_model_call(request, mock_handler)
    assert isinstance(response, ModelResponse)
    assert call_count == 0
