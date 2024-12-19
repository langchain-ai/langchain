import threading
from typing import Any, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

MODEL_COST_PER_1K_INPUT_TOKENS = {
    "anthropic.claude-instant-v1": 0.0008,
    "anthropic.claude-v2": 0.008,
    "anthropic.claude-v2:1": 0.008,
    "anthropic.claude-3-sonnet-20240229-v1:0": 0.003,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 0.003,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 0.003,
    "anthropic.claude-3-haiku-20240307-v1:0": 0.00025,
    "anthropic.claude-3-opus-20240229-v1:0": 0.015,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 0.0008,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    "anthropic.claude-instant-v1": 0.0024,
    "anthropic.claude-v2": 0.024,
    "anthropic.claude-v2:1": 0.024,
    "anthropic.claude-3-sonnet-20240229-v1:0": 0.015,
    "anthropic.claude-3-5-sonnet-20240620-v1:0": 0.015,
    "anthropic.claude-3-5-sonnet-20241022-v2:0": 0.015,
    "anthropic.claude-3-haiku-20240307-v1:0": 0.00125,
    "anthropic.claude-3-opus-20240229-v1:0": 0.075,
    "anthropic.claude-3-5-haiku-20241022-v1:0": 0.004,
}


def _get_anthropic_claude_token_cost(
    prompt_tokens: int, completion_tokens: int, model_id: Union[str, None]
) -> float:
    if model_id:
        # The model ID can be a cross-region (system-defined) inference profile ID,
        # which has a prefix indicating the region (e.g., 'us', 'eu') but
        # shares the same token costs as the "base model".
        # By extracting the "base model ID", by taking the last two segments
        # of the model ID, we can map cross-region inference profile IDs to
        # their corresponding cost entries.
        base_model_id = model_id.split(".")[-2] + "." + model_id.split(".")[-1]
    else:
        base_model_id = None
    """Get the cost of tokens for the Claude model."""
    if base_model_id not in MODEL_COST_PER_1K_INPUT_TOKENS:
        raise ValueError(
            f"Unknown model: {model_id}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_INPUT_TOKENS.keys())
        )
    return (prompt_tokens / 1000) * MODEL_COST_PER_1K_INPUT_TOKENS[base_model_id] + (
        completion_tokens / 1000
    ) * MODEL_COST_PER_1K_OUTPUT_TOKENS[base_model_id]


class BedrockAnthropicTokenUsageCallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks bedrock anthropic info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    successful_requests: int = 0
    total_cost: float = 0.0

    def __init__(self) -> None:
        super().__init__()
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (
            f"Tokens Used: {self.total_tokens}\n"
            f"\tPrompt Tokens: {self.prompt_tokens}\n"
            f"\tCompletion Tokens: {self.completion_tokens}\n"
            f"Successful Requests: {self.successful_requests}\n"
            f"Total Cost (USD): ${self.total_cost}"
        )

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Print out the prompts."""
        pass

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Print out the token."""
        pass

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Collect token usage."""
        if response.llm_output is None:
            return None

        if "usage" not in response.llm_output:
            with self._lock:
                self.successful_requests += 1
            return None

        # compute tokens and cost for this request
        token_usage = response.llm_output["usage"]
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        model_id = response.llm_output.get("model_id", None)
        total_cost = _get_anthropic_claude_token_cost(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_id=model_id,
        )

        # update shared state behind lock
        with self._lock:
            self.total_cost += total_cost
            self.total_tokens += total_tokens
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "BedrockAnthropicTokenUsageCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "BedrockAnthropicTokenUsageCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
