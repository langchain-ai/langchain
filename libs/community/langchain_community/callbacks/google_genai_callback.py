import threading
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

MODEL_COST_PER_1K_INPUT_TOKENS = {
    "gemini-2.0-flash": lambda token_count: 0.0001,
    "gemini-2.0-flash-001": lambda token_count: 0.0001,
    "gemini-2.0-flash-lite": lambda token_count: 0.000075,
    "gemini-2.0-flash-lite-001": lambda token_count: 0.000075,
    "gemini-1.5-pro": lambda token_count: 0.00125 if token_count <= 128000 else 0.005,
    "gemini-1.5-pro-002": lambda token_count: 0.00125
    if token_count <= 128000
    else 0.005,
    "gemini-1.5-flash": lambda token_count: 0.000075
    if token_count <= 128000
    else 0.00015,
    "gemini-1.5-flash-002": lambda token_count: 0.000075
    if token_count <= 128000
    else 0.00015,
    "gemini-1.5-flash-8b": lambda token_count: 0.0000375
    if token_count <= 128000
    else 0.000075,
    "gemini-1.5-flash-8b-001": lambda token_count: 0.0000375
    if token_count <= 128000
    else 0.000075,
}

MODEL_COST_PER_1K_INPUT_CACHING_TOKENS = {
    "gemini-2.0-flash": lambda token_count: 0.000025,
    "gemini-2.0-flash-001": lambda token_count: 0.000025,
    "gemini-1.5-pro": lambda token_count: 0.0003125
    if token_count <= 128000
    else 0.000625,
    "gemini-1.5-pro-002": lambda token_count: 0.0003125
    if token_count <= 128000
    else 0.000625,
    "gemini-1.5-flash": lambda token_count: 0.00001875
    if token_count <= 128000
    else 0.0000375,
    "gemini-1.5-flash-002": lambda token_count: 0.00001875
    if token_count <= 128000
    else 0.0000375,
    "gemini-1.5-flash-8b": lambda token_count: 0.00001
    if token_count <= 128000
    else 0.00002,
    "gemini-1.5-flash-8b-001": lambda token_count: 0.00001
    if token_count <= 128000
    else 0.00002,
}

MODEL_COST_PER_1K_OUTPUT_TOKENS = {
    "gemini-2.0-flash": lambda token_count: 0.0004,
    "gemini-2.0-flash-001": lambda token_count: 0.0004,
    "gemini-2.0-flash-lite": lambda token_count: 0.0003,
    "gemini-2.0-flash-lite-001": lambda token_count: 0.0003,
    "gemini-1.5-pro": lambda token_count: 0.0025 if token_count <= 128000 else 0.01,
    "gemini-1.5-pro-002": lambda token_count: 0.0025 if token_count <= 128000 else 0.01,
    "gemini-1.5-flash": lambda token_count: 0.0003 if token_count <= 128000 else 0.0006,
    "gemini-1.5-flash-002": lambda token_count: 0.0003
    if token_count <= 128000
    else 0.0006,
    "gemini-1.5-flash-8b": lambda token_count: 0.00015
    if token_count <= 128000
    else 0.0003,
}


def _get_google_genai_token_cost_for_model(
    model_name: str,
    total_tokens: int,
    input_tokens: int,
    input_cached_tokens: int,
    output_tokens: int,
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        total_tokens: Total number of tokens.
        input_tokens: Number of input tokens.
        input_cached_tokens: Number of input cached tokens.
        output_tokens: Number of output tokens.

    Returns:
        Cost in USD.
    """
    if model_name not in MODEL_COST_PER_1K_INPUT_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid Google GenAI model."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_INPUT_TOKENS.keys())
        )
    if (
        input_cached_tokens > 0
        and model_name not in MODEL_COST_PER_1K_INPUT_CACHING_TOKENS
    ):
        raise ValueError(
            f"Model {model_name} does not support input token caching."
            "Known models are: "
            + ", ".join(MODEL_COST_PER_1K_INPUT_CACHING_TOKENS.keys())
        )
    if model_name not in MODEL_COST_PER_1K_OUTPUT_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid Google GenAI model."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_OUTPUT_TOKENS.keys())
        )

    cost = MODEL_COST_PER_1K_INPUT_TOKENS[model_name](total_tokens) * (
        input_tokens / 1000
    )
    if input_cached_tokens > 0:
        cost += MODEL_COST_PER_1K_INPUT_CACHING_TOKENS[model_name](total_tokens) * (
            input_cached_tokens / 1000
        )
    cost += MODEL_COST_PER_1K_OUTPUT_TOKENS[model_name](total_tokens) * (
        output_tokens / 1000
    )
    return cost


class GoogleGenAICallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks OpenAI info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    prompt_tokens_cached: int = 0
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
            f"\t\tPrompt Tokens Cached: {self.prompt_tokens_cached}\n"
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
        # Check for usage_metadata (langchain-core >= 0.2.2)
        try:
            generation = response.generations[0][0]
        except IndexError:
            generation = None
        if isinstance(generation, ChatGeneration):
            try:
                message = generation.message
                if isinstance(message, AIMessage):
                    usage_metadata = message.usage_metadata
                    response_metadata = message.response_metadata
                else:
                    usage_metadata = None
                    response_metadata = None
            except AttributeError:
                usage_metadata = None
                response_metadata = None
        else:
            usage_metadata = None
            response_metadata = None

        prompt_tokens_cached = 0

        if response_model_name := (response_metadata or {}).get("model_name"):
            model_name = response_model_name
        elif response.llm_output is None:
            model_name = ""
        else:
            model_name = response.llm_output.get("model_name", "")

        if usage_metadata is not None:
            total_tokens = usage_metadata["total_tokens"]
            prompt_tokens = usage_metadata["output_tokens"]
            completion_tokens = usage_metadata["input_tokens"]
            if "cache_read" in usage_metadata.get("input_token_details", {}):
                prompt_tokens_cached = usage_metadata["input_token_details"][
                    "cache_read"
                ]

        total_cost = _get_google_genai_token_cost_for_model(
            model_name,
            total_tokens,
            prompt_tokens,
            prompt_tokens_cached,
            completion_tokens,
        )

        # update shared state behind lock
        with self._lock:
            self.total_cost += total_cost
            self.total_tokens += total_tokens
            self.prompt_tokens += prompt_tokens
            self.prompt_tokens_cached += prompt_tokens_cached
            self.completion_tokens += completion_tokens
            self.successful_requests += 1

    def __copy__(self) -> "GoogleGenAICallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "GoogleGenAICallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
