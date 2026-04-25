"""Intent-aware model routing middleware for agents."""

from __future__ import annotations

from typing import TYPE_CHECKING

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

    from langchain_core.language_models.chat_models import BaseChatModel
    from langchain_core.messages import AIMessage, BaseMessage


class IntentAwareModelRouterMiddleware(
    AgentMiddleware[AgentState[ResponseT], ContextT, ResponseT]
):
    """Route model calls to fast, balanced, or quality model by request intent.

    The middleware inspects user-facing message content and applies a simple policy:

    - High-stakes intents (e.g., legal/medical/financial) -> quality model
    - Clearly simple tasks (summarize/extract/rewrite/classify, short prompt) -> fast model
    - Everything else -> balanced model

    It also annotates the resulting AI message with a routing trace in `response_metadata`.
    """

    _HIGH_STAKES_KEYWORDS = (
        "legal",
        "medical",
        "diagnosis",
        "financial",
        "investment",
        "tax",
        "compliance",
        "contract",
        "security vulnerability",
        "production incident",
    )
    _SIMPLE_TASK_KEYWORDS = (
        "summarize",
        "summary",
        "rewrite",
        "rephrase",
        "extract",
        "classify",
        "translate",
        "proofread",
    )

    def __init__(
        self,
        *,
        fast_model: str | BaseChatModel,
        balanced_model: str | BaseChatModel,
        quality_model: str | BaseChatModel,
        short_prompt_char_limit: int = 600,
        policy_version: str = "v1",
    ) -> None:
        """Initialize routing middleware with model tiers."""
        super().__init__()
        self.fast_model = self._coerce_model(fast_model)
        self.balanced_model = self._coerce_model(balanced_model)
        self.quality_model = self._coerce_model(quality_model)
        self.short_prompt_char_limit = short_prompt_char_limit
        self.policy_version = policy_version

    @staticmethod
    def _coerce_model(model: str | BaseChatModel) -> BaseChatModel:
        if isinstance(model, str):
            return init_chat_model(model)
        return model

    @staticmethod
    def _extract_message_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
            return " ".join(parts)
        return ""

    def _combined_prompt_text(self, request: ModelRequest[ContextT]) -> str:
        return " ".join(self._extract_message_text(msg) for msg in request.messages)

    def _choose_model(
        self, request: ModelRequest[ContextT]
    ) -> tuple[BaseChatModel, str, dict[str, object]]:
        prompt_text = self._combined_prompt_text(request).lower()
        prompt_length = len(prompt_text)

        high_stakes_hits = [k for k in self._HIGH_STAKES_KEYWORDS if k in prompt_text]
        simple_task_hits = [k for k in self._SIMPLE_TASK_KEYWORDS if k in prompt_text]
        is_short_prompt = prompt_length <= self.short_prompt_char_limit

        if high_stakes_hits:
            return (
                self.quality_model,
                "high_stakes",
                {
                    "high_stakes_hits": high_stakes_hits,
                    "simple_task_hits": simple_task_hits,
                    "prompt_length": prompt_length,
                    "is_short_prompt": is_short_prompt,
                },
            )
        if simple_task_hits and is_short_prompt:
            return (
                self.fast_model,
                "simple_short_task",
                {
                    "high_stakes_hits": high_stakes_hits,
                    "simple_task_hits": simple_task_hits,
                    "prompt_length": prompt_length,
                    "is_short_prompt": is_short_prompt,
                },
            )
        return (
            self.balanced_model,
            "default_balanced",
            {
                "high_stakes_hits": high_stakes_hits,
                "simple_task_hits": simple_task_hits,
                "prompt_length": prompt_length,
                "is_short_prompt": is_short_prompt,
            },
        )

    def _annotate_routing_trace(
        self,
        response: ModelResponse[ResponseT],
        reason: str,
        decision_features: dict[str, object],
    ) -> None:
        first_message = response.result[0] if response.result else None
        if first_message is None or not hasattr(first_message, "response_metadata"):
            return

        metadata = getattr(first_message, "response_metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            first_message.response_metadata = metadata  # type: ignore[attr-defined]

        metadata["routing"] = {
            "policy": "intent_aware_router",
            "policy_version": self.policy_version,
            "route_reason": reason,
            "decision_features": decision_features,
        }

    def wrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], ModelResponse[ResponseT]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        model, reason, features = self._choose_model(request)
        response = handler(request.override(model=model))
        if isinstance(response, ModelResponse):
            self._annotate_routing_trace(response, reason, features)
        return response

    async def awrap_model_call(
        self,
        request: ModelRequest[ContextT],
        handler: Callable[[ModelRequest[ContextT]], Awaitable[ModelResponse[ResponseT]]],
    ) -> ModelResponse[ResponseT] | AIMessage:
        model, reason, features = self._choose_model(request)
        response = await handler(request.override(model=model))
        if isinstance(response, ModelResponse):
            self._annotate_routing_trace(response, reason, features)
        return response
