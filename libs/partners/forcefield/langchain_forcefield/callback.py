"""ForceField callback handler for LangChain.

Scans prompts before they reach the LLM and moderates outputs after generation.

Usage::

    from langchain_openai import ChatOpenAI
    from langchain_forcefield import ForceFieldCallbackHandler

    handler = ForceFieldCallbackHandler(sensitivity="high")
    llm = ChatOpenAI(callbacks=[handler])
    llm.invoke("Hello")
"""

from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from forcefield import Guard

logger = logging.getLogger(__name__)


class PromptBlockedError(Exception):
    """Raised when ForceField blocks a prompt in the LangChain pipeline."""

    def __init__(self, message: str, scan_result: Any = None) -> None:
        super().__init__(message)
        self.scan_result = scan_result


class ForceFieldCallbackHandler(BaseCallbackHandler):
    """LangChain callback that scans prompts and moderates outputs.

    Args:
        sensitivity: Detection sensitivity level (low, medium, high, critical).
        block_on_input: Raise PromptBlockedError if input is blocked.
        moderate_output: Run output moderation on LLM responses.
        on_block: Optional callable for custom block handling.
    """

    def __init__(
        self,
        sensitivity: str = "medium",
        block_on_input: bool = True,
        moderate_output: bool = True,
        on_block: Optional[Any] = None,
        **guard_kwargs: Any,
    ) -> None:
        self.guard = Guard(sensitivity=sensitivity, **guard_kwargs)
        self.block_on_input = block_on_input
        self.moderate_output = moderate_output
        self.on_block = on_block
        self._last_result: Any = None

    @property
    def last_result(self) -> Any:
        """The most recent scan result."""
        return self._last_result

    def on_llm_start(
        self,
        serialized: dict[str, Any],
        prompts: list[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Scan prompts before they reach the LLM."""
        for prompt in prompts:
            result = self.guard.scan(prompt)
            self._last_result = result
            if result.blocked:
                logger.warning(
                    "ForceField blocked prompt: risk=%.2f rules=%s",
                    result.risk_score,
                    result.rules_triggered,
                )
                if self.on_block:
                    self.on_block(result)
                if self.block_on_input:
                    raise PromptBlockedError(
                        f"ForceField blocked: {', '.join(result.rules_triggered)}",
                        scan_result=result,
                    )

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[list[str]] = None,
        metadata: Optional[dict[str, Any]] = None,
        **kwargs: Any,
    ) -> None:
        """Scan chat messages before they reach the LLM."""
        for message_batch in messages:
            for msg in message_batch:
                content = getattr(msg, "content", None) or ""
                role = getattr(msg, "type", "unknown")
                if role in ("human", "tool") and content:
                    result = self.guard.scan(content)
                    self._last_result = result
                    if result.blocked:
                        logger.warning(
                            "ForceField blocked chat message: risk=%.2f rules=%s",
                            result.risk_score,
                            result.rules_triggered,
                        )
                        if self.on_block:
                            self.on_block(result)
                        if self.block_on_input:
                            raise PromptBlockedError(
                                f"ForceField blocked: {', '.join(result.rules_triggered)}",
                                scan_result=result,
                            )

    def on_llm_end(
        self,
        response: Any,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> None:
        """Moderate LLM outputs for harmful content."""
        if not self.moderate_output:
            return
        try:
            for gen_list in response.generations:
                for gen in gen_list:
                    text = gen.text if hasattr(gen, "text") else str(gen)
                    if text:
                        mod = self.guard.moderate(text)
                        if not mod.passed:
                            logger.warning(
                                "ForceField moderation flagged output: action=%s categories=%s",
                                mod.action.value,
                                mod.categories,
                            )
        except Exception:
            pass

    def on_llm_error(
        self, error: BaseException, **kwargs: Any
    ) -> None:
        """Handle LLM errors (no-op)."""
