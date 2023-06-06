"""Callback Handler streams to stdout on new llm token."""
import sys
from typing import Any, Dict, List, Optional

from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

DEFAULT_ANSWER_PREFIX_TOKENS = ["\nFinal", " Answer", ":"]


class FinalStreamingStdOutCallbackHandler(StreamingStdOutCallbackHandler):
    """Callback handler for streaming in agents.
    Only works with agents using LLMs that support streaming.

    Only the final output of the agent will be streamed.
    """

    def __init__(self, answer_prefix_tokens: Optional[List[str]] = None) -> None:
        super().__init__()
        if answer_prefix_tokens is None:
            answer_prefix_tokens = DEFAULT_ANSWER_PREFIX_TOKENS
        self.answer_prefix_tokens = answer_prefix_tokens
        self.last_tokens = [""] * len(answer_prefix_tokens)
        self.answer_reached = False

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        self.answer_reached = False

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""

        # Remember the last n tokens, where n = len(answer_prefix_tokens)
        self.last_tokens.append(token)
        if len(self.last_tokens) > len(self.answer_prefix_tokens):
            self.last_tokens.pop(0)

        # Check if the last n tokens match the answer_prefix_tokens list ...
        if self.last_tokens == self.answer_prefix_tokens:
            self.answer_reached = True
            # Do not print the last token in answer_prefix_tokens,
            # as it's not part of the answer yet
            return

        # ... if yes, then print tokens from now on
        if self.answer_reached:
            sys.stdout.write(token)
            sys.stdout.flush()
