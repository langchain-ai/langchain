"""Ratelimiting Handler to limit requests or tokens"""

import logging
from typing import Any, Dict, List, Literal, Optional

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

logger = logging.getLogger(__name__)
try:
    from upstash_ratelimit import Ratelimit
except ImportError:
    Ratelimit = None


class UpstashRatelimitError(Exception):
    """
    Upstash Ratelimit Error

    Raised when the rate limit is reached in `UpstashRatelimitHandler`
    """

    def __init__(
        self,
        message: str,
        type: Literal["token", "request"],
        limit: Optional[int] = None,
        reset: Optional[float] = None,
    ):
        """
        Args:
            message (str): error message
            type (str): The kind of the limit which was reached. One of
                "token" or "request"
            limit (Optional[int]): The limit which was reached. Passed when type
                is request
            reset (Optional[int]): unix timestamp in milliseconds when the limits
                are reset. Passed when type is request
        """
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
        self.type = type
        self.limit = limit
        self.reset = reset


class UpstashRatelimitHandler(BaseCallbackHandler):
    """
    Callback to handle rate limiting based on the number of requests
    or the number of tokens in the input.

    It uses Upstash Ratelimit to track the ratelimit which utilizes
    Upstash Redis to track the state.

    Should not be passed to the chain when initialising the chain.
    This is because the handler has a state which should be fresh
    every time invoke is called. Instead, initialise and pass a handler
    every time you invoke.
    """

    raise_error = True
    _checked: bool = False

    def __init__(
        self,
        identifier: str,
        *,
        token_ratelimit: Optional[Ratelimit] = None,
        request_ratelimit: Optional[Ratelimit] = None,
        include_output_tokens: bool = False,
    ):
        """
        Creates UpstashRatelimitHandler. Must be passed an identifier to
        ratelimit like a user id or an ip address.

        Additionally, it must be passed at least one of token_ratelimit
        or request_ratelimit parameters.

        Args:
            identifier Union[int, str]: the identifier
            token_ratelimit Optional[Ratelimit]: Ratelimit to limit the
                number of tokens. Only works with OpenAI models since only
                these models provide the number of tokens as information
                in their output.
            request_ratelimit Optional[Ratelimit]: Ratelimit to limit the
                number of requests
            include_output_tokens bool: Whether to count output tokens when
                rate limiting based on number of tokens. Only used when
                `token_ratelimit` is passed. False by default.

        Example:
            .. code-block:: python

                from upstash_redis import Redis
                from upstash_ratelimit import Ratelimit, FixedWindow

                redis = Redis.from_env()
                ratelimit = Ratelimit(
                    redis=redis,
                    # fixed window to allow 10 requests every 10 seconds:
                    limiter=FixedWindow(max_requests=10, window=10),
                )

                user_id = "foo"
                handler = UpstashRatelimitHandler(
                    identifier=user_id,
                    request_ratelimit=ratelimit
                )

                # Initialize a simple runnable to test
                chain = RunnableLambda(str)

                # pass handler as callback:
                output = chain.invoke(
                    "input",
                    config={
                        "callbacks": [handler]
                    }
                )

        """
        if not any([token_ratelimit, request_ratelimit]):
            raise ValueError(
                "You must pass at least one of input_token_ratelimit or"
                " request_ratelimit parameters for handler to work."
            )

        self.identifier = identifier
        self.token_ratelimit = token_ratelimit
        self.request_ratelimit = request_ratelimit
        self.include_output_tokens = include_output_tokens

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> Any:
        """
        Run when chain starts running.

        on_chain_start runs multiple times during a chain execution. To make
        sure that it's only called once, we keep a bool state `_checked`. If
        not `self._checked`, we call limit with `request_ratelimit` and raise
        `UpstashRatelimitError` if the identifier is rate limited.
        """
        if self.request_ratelimit and not self._checked:
            response = self.request_ratelimit.limit(self.identifier)
            if not response.allowed:
                raise UpstashRatelimitError(
                    "Request limit reached!", "request", response.limit, response.reset
                )
            self._checked = True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """
        Run when LLM starts running
        """
        if self.token_ratelimit:
            remaining = self.token_ratelimit.get_remaining(self.identifier)
            if remaining <= 0:
                raise UpstashRatelimitError("Token limit reached!", "token")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """
        Run when LLM ends running

        If the `include_output_tokens` is set to True, number of tokens
        in LLM completion are counted for rate limiting
        """
        if self.token_ratelimit:
            try:
                llm_output = response.llm_output or {}
                token_usage = llm_output["token_usage"]
                token_count = (
                    token_usage["total_tokens"]
                    if self.include_output_tokens
                    else token_usage["prompt_tokens"]
                )
            except KeyError:
                raise ValueError(
                    "LLM response doesn't include"
                    " `token_usage: {total_tokens: int, prompt_tokens: int}`"
                    "  field. To use UpstashRatelimitHandler with token_ratelimit,"
                    " either use a model which returns token_usage (like "
                    " OpenAI models) or rate limit only with request_ratelimit."
                )

            # call limit to add the completion tokens to rate limit
            # but don't raise exception since we already generated
            # the tokens and would rather continue execution.
            self.token_ratelimit.limit(self.identifier, rate=token_count)

    def reset(self, identifier: Optional[str] = None) -> "UpstashRatelimitHandler":
        """
        Creates a new UpstashRatelimitHandler object with the same
        ratelimit configurations but with a new identifier if it's
        provided.

        Also resets the state of the handler.
        """
        return UpstashRatelimitHandler(
            identifier=identifier or self.identifier,
            token_ratelimit=self.token_ratelimit,
            request_ratelimit=self.request_ratelimit,
            include_output_tokens=self.include_output_tokens,
        )
