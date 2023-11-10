"""This token usage callback handler can be used for LLMs that does not provide usage info.

It counts the consumed tokens locally.
"""

import datetime
from collections import defaultdict
from typing import Any, Callable, Dict, List
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema.output import ChatGenerationChunk, GenerationChunk, LLMResult

from ..reporters import TokenUsageReport, TokenUsageReporter
from .timer import TokenUsageTimer


class LocalTokenUsageCallbackHandler(BaseCallbackHandler):
    """This token usage callback handler can be used for LLMs that does not provide usage info.

    It counts the consumed tokens locally.
    """

    model_name: str
    caller_id: str
    token_counter_func: Callable[[str], int]
    cost_func: Callable[[int, int], float] | None
    _timers: dict[UUID, TokenUsageTimer]
    _prompt_tokens_by_run: dict[UUID, int]

    def __init__(
        self,
        reporter: TokenUsageReporter,
        model_name: str,
        caller_id: str,
        token_counter_func: Callable[[str], int],
        cost_func: Callable[[int, int], float] | None = None,
    ) -> None:
        """This token usage callback handler can be used for LLMs that does not provide usage info.

        It counts the consumed tokens locally.

        Args:
            reporter (TokenUsageReporter): The reporter that will be used to send the metrics
                to the metrics repository.
            model_name (str): The name of the model being used.
            caller_id (str): Identifies the caller in the metrics repository.
            token_counter_func (Callable[[str], int]): The token counter function specific to the
                model being used. It should return the number of tokens of the text it receives.
            cost_func (Callable[[int, int], float] | None, optional): Optional cost function.
                It will be called with the number of tokens in the prompt and the number of
                generated tokens, and it should return the cost of the run. Defaults to None.
        """
        self.reporter = reporter
        self.model_name = model_name
        self.caller_id = caller_id
        self.token_counter_func = token_counter_func
        self.cost_func = cost_func
        self._prompt_tokens_by_run = defaultdict(int)
        self._timers = defaultdict(TokenUsageTimer)

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
        prompt_tokens = sum(self.token_counter_func(prompt) for prompt in prompts)
        self._prompt_tokens_by_run[run_id] = prompt_tokens

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
        timer = self._timers.pop(run_id, None)
        if timer is not None:
            timer.end()
        timestamp = datetime.datetime.now()
        prompt_tokens = self._prompt_tokens_by_run.pop(run_id, 0)
        completion_tokens = sum(
            self.token_counter_func(gen.text) for gens in response.generations for gen in gens
        )
        total_tokens = prompt_tokens + completion_tokens
        total_cost: float | None = None
        if self.cost_func is not None:
            total_cost = self.cost_func(prompt_tokens, completion_tokens)

        first_token_time = timer.first_token_elapsed if timer is not None else None
        completion_time = timer.completion_elapsed if timer is not None else None

        self.reporter.send_report(
            TokenUsageReport(
                timestamp=timestamp,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                total_cost=total_cost,
                first_token_time=first_token_time,
                completion_time=completion_time,
                model_name=self.model_name,
                caller_id=self.caller_id,
            )
        )
