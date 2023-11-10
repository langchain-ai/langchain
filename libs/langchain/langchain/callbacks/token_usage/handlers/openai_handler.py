"""OpenAI token usage callback handler for LangChain."""

import datetime
import os
from collections import defaultdict
from typing import Any, Dict, List
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.openai_info import get_openai_token_cost_for_model, standardize_model_name
from langchain.schema.output import ChatGenerationChunk, GenerationChunk, LLMResult

from ..reporters import TokenUsageReport, TokenUsageReporter
from .timer import TokenUsageTimer


def _get_caller_id(val: str | None) -> str | None:
    return val[-4:] if val is not None and len(val) >= 4 else None


class OpenAITokenUsageCallbackHandler(BaseCallbackHandler):
    """Collects metrics about the token usage of OpenAI LLM runs."""

    reporter: TokenUsageReporter
    _timers: Dict[UUID, TokenUsageTimer]
    _caller_id: str | None = None

    def __init__(self, reporter: TokenUsageReporter) -> None:
        """Collects metrics about the token usage of OpenAI LLM runs.

        Args:
            reporter (TokenUsageReporter): The reporter that will be used to send the metrics
                to the metrics repository.
        """
        try:
            import openai
        except ImportError as err:
            raise ImportError(
                "openai package not found, please install with `pip install openai`"
            ) from err
        self.reporter = reporter
        self._timers = defaultdict(TokenUsageTimer)
        self._caller_id = _get_caller_id(openai.api_key)
        if self._caller_id is None:
            self._caller_id = _get_caller_id(os.environ.get("OPENAI_API_KEY"))

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: List[str] | None = None,
        metadata: Dict[str, Any] | None = None,
        **kwargs: Any
    ) -> None:
        """Called when the LLM starts processing the request."""
        self._timers[run_id].start()

    def on_llm_new_token(
        self,
        token: str,
        *,
        chunk: GenerationChunk | ChatGenerationChunk | None = None,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any
    ) -> Any:
        """Called when the LLM emits a new token."""
        self._timers[run_id].new_token()

    def on_llm_end(
        self, response: LLMResult, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any
    ) -> None:
        """Called when the LLM finishes processing the request."""
        timer = self._timers.pop(run_id)
        timer.end()
        timestamp = datetime.datetime.now()

        # get stats
        if response.llm_output is None:
            return None
        if "token_usage" not in response.llm_output:
            return None
        token_usage = response.llm_output["token_usage"]
        completion_tokens = token_usage.get("completion_tokens")
        prompt_tokens = token_usage.get("prompt_tokens")
        model_name = standardize_model_name(response.llm_output.get("model_name", ""))
        total_cost: float | None = None
        try:
            completion_cost = (
                get_openai_token_cost_for_model(model_name, completion_tokens, is_completion=True)
                if completion_tokens is not None
                else 0.0
            )
            prompt_cost = (
                get_openai_token_cost_for_model(model_name, prompt_tokens)
                if prompt_tokens is not None
                else 0.0
            )
            total_cost = prompt_cost + completion_cost
        except ValueError:
            pass
        total_tokens: int | None = token_usage.get("total_tokens")

        self.reporter.send_report(
            TokenUsageReport(
                timestamp=timestamp,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_cost=total_cost,
                first_token_time=timer.first_token_elapsed,
                completion_time=timer.completion_elapsed,
                model_name=model_name,
                caller_id=self._caller_id,
            )
        )
