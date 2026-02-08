from typing import Any, ClassVar, cast

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain.agents.middleware.auto_model_selector import (
    ComplexityLevel,
    LLMAutoModelSelector,
)
from langchain.agents.middleware.types import ModelRequest, ModelResponse


class MockChatModel(BaseChatModel):
    """Mock Chat Model for testing."""

    response: str = "mock response"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.response))])

    @property
    def _llm_type(self) -> str:
        return "mock-chat-model"


class MockClassifierModel(BaseChatModel):
    """Mock Classifier Model that returns predefined complexity."""

    complexity_map: ClassVar[dict[str, str]] = {}
    default_complexity: str = "medium"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Simple logic: check if the prompt contains certain keywords if passed?
        # Or just return a complexity if we set it up that way.
        # But here we act as a model that READS the prompt.
        # The middleware sends: SystemMessage(...), AIMessage("Conversation History: ...")

        # For simplicity, we can inspect the content of the last message
        last_msg = messages[-1].content
        if isinstance(last_msg, str):
            if "hard" in last_msg:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="hard"))])
            if "easy" in last_msg:
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content="easy"))])

        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=self.default_complexity))]
        )

    @property
    def _llm_type(self) -> str:
        return "mock-classifier"


def test_auto_model_selector_sync() -> None:
    # Setup
    easy_model = MockChatModel(response="response from easy model")
    medium_model = MockChatModel(response="response from medium model")
    hard_model = MockChatModel(response="response from hard model")

    models: dict[ComplexityLevel, str | BaseChatModel] = {
        cast("ComplexityLevel", "easy"): easy_model,
        cast("ComplexityLevel", "medium"): medium_model,
        cast("ComplexityLevel", "hard"): hard_model,
    }

    classifier = MockClassifierModel()

    selector: LLMAutoModelSelector = LLMAutoModelSelector(
        models=models, classifier_model=classifier
    )

    # Test Easy case
    # We simulate complexity by putting "easy" in the conversation history
    # which the classifier reads
    request = ModelRequest(
        messages=[HumanMessage(content="This is an easy task")],
        model=medium_model,
    )

    def handler(req: ModelRequest) -> ModelResponse:
        # The handler invokes the model in the request
        model = req.model
        # We assume req.model is set by middleware
        if model:
            response = model.invoke(req.messages)
            return ModelResponse(result=[response])
        return ModelResponse(result=[AIMessage(content="default")])

    result = selector.wrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "response from easy model"

    # Test Hard case
    request_hard = ModelRequest(
        messages=[HumanMessage(content="This is a hard task")], model=medium_model
    )
    result_hard = selector.wrap_model_call(request_hard, handler)
    assert result_hard.result[0].content == "response from hard model"


@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_auto_model_selector_async(anyio_backend: str) -> None:
    # Setup
    easy_model = MockChatModel(response="response from easy model")
    medium_model = MockChatModel(response="response from medium model")
    hard_model = MockChatModel(response="response from hard model")

    models: dict[ComplexityLevel, str | BaseChatModel] = {
        cast("ComplexityLevel", "easy"): easy_model,
        cast("ComplexityLevel", "medium"): medium_model,
        cast("ComplexityLevel", "hard"): hard_model,
    }

    classifier = MockClassifierModel()

    selector: LLMAutoModelSelector = LLMAutoModelSelector(
        models=models, classifier_model=classifier
    )

    # Test Medium case (default)
    request = ModelRequest(
        messages=[HumanMessage(content="This is a normal task")], model=medium_model
    )

    async def handler(req: ModelRequest) -> ModelResponse:
        model = req.model
        if model:
            response = await model.ainvoke(req.messages)
            return ModelResponse(result=[response])
        return ModelResponse(result=[AIMessage(content="default")])

    result = await selector.awrap_model_call(request, handler)
    assert isinstance(result, ModelResponse)
    assert result.result[0].content == "response from medium model"
