import re
import threading
from typing import Any, Dict, List, Union

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

MODEL_COST_PER_1K_INPUT_TOKENS = {
    # Premier models
    "mistral-large-latest": 0.002,
    "pixtral-large-latest": 0.002,
    "mistral-saba-latest": 0.0002,
    "codestral-latest": 0.0003,
    "ministral-8b-latest": 0.0001,
    "ministral-3b-latest": 0.00004,
    "mistral-embed": 0.0001,
    "mistral-moderation-latest": 0.0001,
    # free models
    "pixtral-12b": 0.00015,
    "mistral-small-latest": 0.0001,
    "mistral-nemo": 0.00015,
    "open-mistral-7b": 0.00025,
    "open-mixtral-8x7b": 0.0007,
    "open-mixtral-8x22b": 0.002,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    # Premium models
    "mistral-large-latest": 0.006,
    "pixtral-large-latest": 0.006,
    "mistral-saba-latest": 0.0006,
    "codestral-latest": 0.0009,
    "ministral-8b-latest": 0.0001,
    "ministral-3b-latest": 0.0004,
    "mistral-embed": 0.00,
    "mistral-moderation-latest": 0.00,
    # Free models
    "pixtral-12b": 0.00015,
    "mistral-small-latest": 0.0003,
    "mistral-nemo": 0.00015,
    "open-mistral-7b": 0.00025,
    "open-mixtral-8x7b": 0.0007,
    "open-mixtral-8x22b": 0.006,
}


def _get_mistral_ai_token_cost(
    prompt_tokens: int, completion_tokens: int, model_id: Union[str, None]
) -> float:
    if model_id:
        # convert numeric version to latest.
        model = re.sub(r"-(\d){4}", "-latest", model_id)
    else:
        model = None
    if model not in MODEL_COST_PER_1K_INPUT_TOKENS:
        raise ValueError(
            f"Unknown model: {model}. Please provide a valid Anthropic model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_INPUT_TOKENS.keys())
        )
    return (prompt_tokens / 1000) * MODEL_COST_PER_1K_INPUT_TOKENS[model] + (
        completion_tokens / 1000
    ) * MODEL_COST_PER_1K_OUTPUT_TOKENS[model]


class MistralAiCallbackHandler(BaseCallbackHandler):
    """Callback handler that tracks MistralAI info."""

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
        if "token_usage" not in response.llm_output:
            with self._lock:
                self.successful_requests += 1
            return None
        # compute tokens and cost for this request
        token_usage = response.llm_output["token_usage"]
        completion_tokens = token_usage.get("completion_tokens", 0)
        prompt_tokens = token_usage.get("prompt_tokens", 0)
        total_tokens = token_usage.get("total_tokens", 0)
        model_id = response.llm_output.get("model", None)
        total_cost = _get_mistral_ai_token_cost(
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

    def __copy__(self) -> "MistralAiCallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "MistralAiCallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
