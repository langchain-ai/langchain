"""Auto Model Selector Middleware."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal, cast

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ContextT,
    ModelRequest,
    ModelResponse,
    ResponseT,
)
from langchain.chat_models import init_chat_model

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

ComplexityLevel = Literal["easy", "medium", "hard"]

DEFAULT_CLASSIFIER_PROMPT = (
    "You are an expert AI task complexity classifier. "
    "Analyze the following conversation history and determine the complexity level "
    "of the required response. "
    "Respond with ONLY one of the following words: 'easy', 'medium', or 'hard'.\n\n"
    "Guidelines:\n"
    "- 'easy': Simple factual questions, greetings, or requests requiring no tools or simple retrieval.\n"
    "- 'medium': Tasks requiring some reasoning, clarification, or single tool calls.\n"
    "- 'hard': Complex multi-step reasoning, coding tasks, or scenarios requiring multiple tool calls or deep analysis."
)


class LLMAutoModelSelector(AgentMiddleware[AgentState[ResponseT], ContextT]):
    """Middleware that dynamically selects an LLM based on task complexity.

    It uses a classifier model to evaluate the conversation history and pick
    the most appropriate model from a provided set of options.
    """

    models: dict[ComplexityLevel, BaseChatModel]
    classifier_model: BaseChatModel

    def __init__(
        self,
        models: dict[ComplexityLevel, str | BaseChatModel],
        classifier_model: str | BaseChatModel,
        *,
        last_k_messages: int = 5,
        system_prompt: str = DEFAULT_CLASSIFIER_PROMPT,
    ) -> None:
        """Initialize the model selector middleware.

        Args:
            models: Mapping of complexity levels to models. Each value can be
                a model identifier string (e.g., "openai:gpt-4o-mini") or a
                BaseChatModel instance. All three levels (easy, medium, hard)
                should be provided.
            classifier_model: Model used to classify task complexity.
            last_k_messages: Number of recent messages to analyze for complexity
                classification.
            system_prompt: Custom instructions for the classifier LLM.
        """
        super().__init__()
        self.models = {}
        self.classifier_model = classifier_model
        self.last_k_messages = last_k_messages
        self.system_prompt = system_prompt
        self.tools = []  # No tools registered by this middleware

        # Initialize chat model instances if strings are provided
        if isinstance(self.classifier_model, str):
            self.classifier_model = init_chat_model(self.classifier_model)

        for level, model in models.items():
            if isinstance(model, str):
                self.models[cast(ComplexityLevel, level)] = init_chat_model(model)
            else:
                self.models[cast(ComplexityLevel, level)] = model

    def _prepare_classification_messages(
        self, request: ModelRequest
    ) -> list[SystemMessage | AIMessage]:
        """Prepare messages for the classifier."""
        # Get the last k messages
        messages_to_analyze = request.messages[-self.last_k_messages :]

        # Convert messages to a string representation for the classifier
        # We wrap them in a user message context for the classifier
        conversation_history = "\n".join(
            f"{msg.type}: {msg.content}" for msg in messages_to_analyze
        )

        return [
            SystemMessage(content=self.system_prompt),
            AIMessage(content=f"Conversation History:\n{conversation_history}"),
        ]

    def _select_model(self, complexity: str) -> BaseChatModel:
        """Select the model based on complexity."""
        complexity = complexity.strip().lower()
        # Remove any punctuation
        complexity = complexity.strip("'\".,")

        if complexity not in ["easy", "medium", "hard"]:
            logger.warning(
                f"Classifier returned unknown complexity '{complexity}', defaulting to 'medium'"
            )
            return self.models.get("medium", next(iter(self.models.values())))

        # Cast to ComplexityLevel to satisfy type checker
        level = cast(ComplexityLevel, complexity) # type: ignore
        return self.models.get(level, self.models.get("medium", next(iter(self.models.values()))))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse | AIMessage:
        """Intercept model calls to swap in the optimal model (sync version)."""
        if not request.messages:
            # No messages to analyze, use default model from request (or fall back to medium)
            return handler(request)

        try:
            # 1. Prepare classification request
            messages = self._prepare_classification_messages(request)

            # 2. Call classifier
            # We assume classifier_model is a BaseChatModel which has invoke
            response = self.classifier_model.invoke(messages)
            complexity = response.content
            if isinstance(complexity, list):
                 complexity = complexity[0] if complexity else "medium"

            # 3. Select model
            selected_model = self._select_model(str(complexity))

            # 4. Override request
            new_request = request.override(model=selected_model)

            # 5. Delegate to handler
            return handler(new_request)

        except Exception as e:
            logger.error(f"AutoModelSelector failed: {e}. Falling back to default model.")
            return handler(request)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse | AIMessage:
        """Intercept model calls to swap in the optimal model (async version)."""
        if not request.messages:
             return await handler(request)

        try:
            # 1. Prepare classification request
            messages = self._prepare_classification_messages(request)

            # 2. Call classifier
            response = await self.classifier_model.ainvoke(messages)
            complexity = response.content
            if isinstance(complexity, list):
                 complexity = complexity[0] if complexity else "medium"

            # 3. Select model
            selected_model = self._select_model(str(complexity))

            # 4. Override request
            new_request = request.override(model=selected_model)

            # 5. Delegate to handler
            return await handler(new_request)

        except Exception as e:
            logger.error(f"AutoModelSelector failed: {e}. Falling back to default model.")
            return await handler(request)
