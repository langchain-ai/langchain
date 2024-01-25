import threading
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

MODEL_COST_PER_1M_TOKENS = {
    # Taken from https://fireworks.ai/pricing
    "fireworks/mixtral-8x7b-instruct": (0.4, 1.6),
    "fireworks/fw-function-call-34b-v0": (0.7, 2.8),
    "fireworks/qwen-72b-chat": (0.7, 2.8),
    "fireworks/yi-34b-200k-capybara": (0.7, 2.8),
    "fireworks/yi-34b-200k": (0.7, 2.8),
    "fireworks/yi-6b": (0.2, 0.8),
    "fireworks/firellava-13b": (0.2, 0.8),
    "fireworks/mistral-7b-instruct-4k": (0.2, 0.8),
    "fireworks/llama-v2-13b-code-instruct": (0.2, 0.8),
    "fireworks/llama-v2-34b-code-instruct": (0.7, 2.8),
    "fireworks/llama-v2-7b-chat": (0.2, 0.8),
    "fireworks/llama-v2-13b-chat": (0.2, 0.8),
    "fireworks/llama-v2-70b-chat": (0.7, 2.8),
    "fireworks/starcoder-7b-w8a16": (0.2, 0.8),
    "fireworks/starcoder-16b-w8a16": (0.2, 0.8),
    "fireworks/qwen-1-8b-chat": (0.2, 0.8),
    "fireworks/qwen-14b-chat": (0.2, 0.8),
    "fireworks/llamaguard-7b": (0.2, 0.8),
    "fireworks/elyza-japanese-llama-2-7b-fast-instruct": (0.2, 0.8),
    "fireworks/japanese-llava-mistral-7b": (0.2, 0.8),
    "stability/japanese-stablelm-instruct-beta-70b": (0.7, 2.8),
    "stability/japanese-stablelm-instruct-gamma-7b": (0.2, 0.8),
    "stability/japanese-stable-vlm": (0.7, 2.8),
    "fireworks/llama-v2-7b": (0.2, 0.8),
    "fireworks/llava-codellama-34b": (0.7, 2.8),
    "fireworks/llava-v15-13b-fireworks": (0.2, 0.8),
    "fireworks/mistral-7b": (0.2, 0.8),
    "fireworks/mixtral-8x7b": (0.4, 1.6),
    "stability/stablelm-zephyr-3b": (0.2, 0.8),
    "fireworks/zephyr-7b-beta": (0.2, 0.8),
}


def get_model_name(model: str):
    """
    Gets the name of the model from the model string

    Args:
        model: Model string, i.e "accounts/fireworks/models/mistral-7b-instruct-4k"

    Returns:
        The standardized name of the model
    """
    model = model.split("/")
    model_name = f"{model[1]}/{model[3]}"
    if model_name not in MODEL_COST_PER_1M_TOKENS:
        raise ValueError(
            f"Unknown model {model}. Please provide a valid Fireworks model name."
            f"Known models are: {','.join(MODEL_COST_PER_1M_TOKENS.keys())}"
        )
    return model_name


def get_fireworks_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    """
    Gets the cost in USD for a given model and number of tokens

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens
        is_completion: Whether tokens are completion or prompt, default to False

    Returns:
        Cost in USD
    """
    model_cost = MODEL_COST_PER_1M_TOKENS[model_name]
    if is_completion:
        return model_cost[1] * (num_tokens / 1000000)
    else:
        return model_cost[0] * (num_tokens / 1000000)


class FireworksCallbackHandler(BaseCallbackHandler):
    """
    Callback handler that tracks fireworks.ai usage info
    """

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
        """
        Collect token usage data
        """

        if response.llm_output is None:
            return None

        if "token_usage" not in response.llm_output:
            with self._lock:
                self.successful_requests += 1
            return None

        token_usages = response.llm_output["token_usage"]
        total_completion_cost = 0
        total_prompt_cost = 0
        total_completion_tokens = 0
        total_prompt_tokens = 0
        for usage in token_usages:
            # Calculate token cost
            completion_tokens = usage["completion_tokens"]
            prompt_tokens = usage["prompt_tokens"]
            model_name = get_model_name(response.llm_output["model"])
            if model_name in MODEL_COST_PER_1M_TOKENS:
                completion_cost = get_fireworks_token_cost_for_model(
                    model_name, completion_tokens, is_completion=True
                )
                prompt_cost = get_fireworks_token_cost_for_model(
                    model_name, prompt_tokens
                )
            else:
                completion_cost = 0
                prompt_cost = 0
            total_completion_cost += completion_cost
            total_prompt_cost + prompt_cost
            total_completion_tokens += completion_tokens
            total_prompt_tokens += prompt_tokens

        # Update state behind lock
        with self._lock:
            self.total_cost += total_prompt_cost + total_completion_cost
            self.total_tokens += total_completion_tokens + total_prompt_tokens
            self.completion_tokens += total_completion_tokens
            self.prompt_tokens += total_prompt_tokens
            self.successful_requests += 1

    def __copy__(self) -> "FireworksCallbackHandler":
        """
        Returns a copy of the callback handler
        """
        return self

    def __deepcopy__(self, memo: Any) -> "FireworksCallbackHandler":
        """
        Returns a deep copy of the callback handler
        """
        return self
