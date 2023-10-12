from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler
from langchain.schema import LLMResult

DEFAULT_ANSWER_PREFIX_TOKENS = ["Final", "Answer", ":"]


class AsyncFinalIteratorCallbackHandler(AsyncIteratorCallbackHandler):
    """Callback handler that returns an async iterator.
    Only the final output of the agent will be iterated.
    """

    def append_to_last_tokens(self, token: str) -> None:
        self.last_tokens.append(token)
        self.last_tokens_stripped.append(token.strip())
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)
            self.last_tokens_stripped.pop(0)

    def check_if_answer_reached(self) -> bool:
        if self.strip_tokens:
            return self.last_tokens_stripped == self.answer_prefix_tokens_stripped
        else:
            return self.last_tokens == self.answer_prefix_tokens

    def __init__(
        self,
        *,
        answer_prefix_tokens: Optional[List[str]] = None,
        strip_tokens: bool = True,
        stream_prefix: bool = False,
    ) -> None:
        """Instantiate AsyncFinalIteratorCallbackHandler.

        Args:
            answer_prefix_tokens: Token sequence that prefixes the answer.
                Default is ["Final", "Answer", ":"]
            strip_tokens: Ignore white spaces and new lines when comparing
                answer_prefix_tokens to last tokens? (to determine if answer has been
                reached)
            stream_prefix: Should answer prefix itself also be streamed?
        """
        super().__init__()
        if answer_prefix_tokens is None:
            self.answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        else:
            self.answer_prefix_tokens = answer_prefix_tokens
        if strip_tokens:
            self.answer_prefix_tokens_stripped = [
                token.strip() for token in self.answer_prefix_tokens
            ]
        else:
            self.answer_prefix_tokens_stripped = self.answer_prefix_tokens
        self.last_tokens = [""] * len(self.answer_prefix_tokens)
        self.last_tokens_stripped = [""] * len(self.answer_prefix_tokens)
        self.strip_tokens = strip_tokens
        self.stream_prefix = stream_prefix
        self.answer_reached = False

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        # If two calls are made in a row, this resets the state
        self.done.clear()
        self.answer_reached = False

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        if self.answer_reached:
            self.done.set()

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.append_to_last_tokens(token)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.check_if_answer_reached():
            self.answer_reached = True
            if self.stream_prefix:
                for t in self.last_tokens:
                    self.queue.put_nowait(t)
            return

        # If yes, then put tokens from now on
        if self.answer_reached:
            self.queue.put_nowait(token)
