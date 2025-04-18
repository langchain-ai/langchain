"""Callback Handler that prints to std out."""

import threading
from enum import Enum, auto
from typing import Any, Dict, List

from langchain_core._api import warn_deprecated
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

MODEL_COST_PER_1K_TOKENS = {
    # GPT-4.1 input
    "gpt-4.1": 0.002,
    "gpt-4.1-2025-04-14": 0.002,
    "gpt-4.1-cached": 0.0005,
    "gpt-4.1-2025-04-14-cached": 0.0005,
    # GPT-4.1 output
    "gpt-4.1-completion": 0.008,
    "gpt-4.1-2025-04-14-completion": 0.008,
    # GPT-4.1-mini input
    "gpt-4.1-mini": 0.0004,
    "gpt-4.1-mini-2025-04-14": 0.0004,
    "gpt-4.1-mini-cached": 0.0001,
    "gpt-4.1-mini-2025-04-14-cached": 0.0001,
    # GPT-4.1-mini output
    "gpt-4.1-mini-completion": 0.0016,
    "gpt-4.1-mini-2025-04-14-completion": 0.0016,
    # GPT-4.1-nano input
    "gpt-4.1-nano": 0.0001,
    "gpt-4.1-nano-2025-04-14": 0.0001,
    "gpt-4.1-nano-cached": 0.000025,
    "gpt-4.1-nano-2025-04-14-cached": 0.000025,
    # GPT-4.1-nano output
    "gpt-4.1-nano-completion": 0.0004,
    "gpt-4.1-nano-2025-04-14-completion": 0.0004,
    # GPT-4.5-preview input
    "gpt-4.5-preview": 0.075,
    "gpt-4.5-preview-2025-02-27": 0.075,
    "gpt-4.5-preview-cached": 0.0375,
    "gpt-4.5-preview-2025-02-27-cached": 0.0375,
    # GPT-4.5-preview output
    "gpt-4.5-preview-completion": 0.15,
    "gpt-4.5-preview-2025-02-27-completion": 0.15,
    # OpenAI o1 input
    "o1": 0.015,
    "o1-2024-12-17": 0.015,
    "o1-cached": 0.0075,
    "o1-2024-12-17-cached": 0.0075,
    # OpenAI o1 output
    "o1-completion": 0.06,
    "o1-2024-12-17-completion": 0.06,
    # OpenAI o1-pro input
    "o1-pro": 0.15,
    "o1-pro-2025-03-19": 0.15,
    # OpenAI o1-pro output
    "o1-pro-completion": 0.6,
    "o1-pro-2025-03-19-completion": 0.6,
    # OpenAI o3 input
    "o3": 0.01,
    "o3-2025-04-16": 0.01,
    "o3-cached": 0.0025,
    "o3-2025-04-16-cached": 0.0025,
    # OpenAI o3 output
    "o3-completion": 0.04,
    "o3-2025-04-16-completion": 0.04,
    # OpenAI o4-mini input
    "o4-mini": 0.0011,
    "o4-mini-2025-04-16": 0.0011,
    "o4-mini-cached": 0.000275,
    "o4-mini-2025-04-16-cached": 0.000275,
    # OpenAI o4-mini output
    "o4-mini-completion": 0.0044,
    "o4-mini-2025-04-16-completion": 0.0044,
    # OpenAI o3-mini input
    "o3-mini": 0.0011,
    "o3-mini-2025-01-31": 0.0011,
    "o3-mini-cached": 0.00055,
    "o3-mini-2025-01-31-cached": 0.00055,
    # OpenAI o3-mini output
    "o3-mini-completion": 0.0044,
    "o3-mini-2025-01-31-completion": 0.0044,
    # OpenAI o1-mini input (updated pricing)
    "o1-mini": 0.0011,
    "o1-mini-cached": 0.00055,
    "o1-mini-2024-09-12": 0.0011,
    "o1-mini-2024-09-12-cached": 0.00055,
    # OpenAI o1-mini output (updated pricing)
    "o1-mini-completion": 0.0044,
    "o1-mini-2024-09-12-completion": 0.0044,
    # OpenAI o1-preview input
    "o1-preview": 0.015,
    "o1-preview-cached": 0.0075,
    "o1-preview-2024-09-12": 0.015,
    "o1-preview-2024-09-12-cached": 0.0075,
    # OpenAI o1-preview output
    "o1-preview-completion": 0.06,
    "o1-preview-2024-09-12-completion": 0.06,
    # GPT-4o input
    "gpt-4o": 0.0025,
    "gpt-4o-cached": 0.00125,
    "gpt-4o-2024-05-13": 0.005,
    "gpt-4o-2024-08-06": 0.0025,
    "gpt-4o-2024-08-06-cached": 0.00125,
    "gpt-4o-2024-11-20": 0.0025,
    "gpt-4o-2024-11-20-cached": 0.00125,
    # GPT-4o output
    "gpt-4o-completion": 0.01,
    "gpt-4o-2024-05-13-completion": 0.015,
    "gpt-4o-2024-08-06-completion": 0.01,
    "gpt-4o-2024-11-20-completion": 0.01,
    # GPT-4o-audio-preview input
    "gpt-4o-audio-preview": 0.0025,
    "gpt-4o-audio-preview-2024-12-17": 0.0025,
    "gpt-4o-audio-preview-2024-10-01": 0.0025,
    # GPT-4o-audio-preview output
    "gpt-4o-audio-preview-completion": 0.01,
    "gpt-4o-audio-preview-2024-12-17-completion": 0.01,
    "gpt-4o-audio-preview-2024-10-01-completion": 0.01,
    # GPT-4o-realtime-preview input
    "gpt-4o-realtime-preview": 0.005,
    "gpt-4o-realtime-preview-2024-12-17": 0.005,
    "gpt-4o-realtime-preview-2024-10-01": 0.005,
    "gpt-4o-realtime-preview-cached": 0.0025,
    "gpt-4o-realtime-preview-2024-12-17-cached": 0.0025,
    "gpt-4o-realtime-preview-2024-10-01-cached": 0.0025,
    # GPT-4o-realtime-preview output
    "gpt-4o-realtime-preview-completion": 0.02,
    "gpt-4o-realtime-preview-2024-12-17-completion": 0.02,
    "gpt-4o-realtime-preview-2024-10-01-completion": 0.02,
    # GPT-4o-mini input
    "gpt-4o-mini": 0.00015,
    "gpt-4o-mini-cached": 0.000075,
    "gpt-4o-mini-2024-07-18": 0.00015,
    "gpt-4o-mini-2024-07-18-cached": 0.000075,
    # GPT-4o-mini output
    "gpt-4o-mini-completion": 0.0006,
    "gpt-4o-mini-2024-07-18-completion": 0.0006,
    # GPT-4o-mini-audio-preview input
    "gpt-4o-mini-audio-preview": 0.00015,
    "gpt-4o-mini-audio-preview-2024-12-17": 0.00015,
    # GPT-4o-mini-audio-preview output
    "gpt-4o-mini-audio-preview-completion": 0.0006,
    "gpt-4o-mini-audio-preview-2024-12-17-completion": 0.0006,
    # GPT-4o-mini-realtime-preview input
    "gpt-4o-mini-realtime-preview": 0.0006,
    "gpt-4o-mini-realtime-preview-2024-12-17": 0.0006,
    "gpt-4o-mini-realtime-preview-cached": 0.0003,
    "gpt-4o-mini-realtime-preview-2024-12-17-cached": 0.0003,
    # GPT-4o-mini-realtime-preview output
    "gpt-4o-mini-realtime-preview-completion": 0.0024,
    "gpt-4o-mini-realtime-preview-2024-12-17-completion": 0.0024,
    # GPT-4o-mini-search-preview input
    "gpt-4o-mini-search-preview": 0.00015,
    "gpt-4o-mini-search-preview-2025-03-11": 0.00015,
    # GPT-4o-mini-search-preview output
    "gpt-4o-mini-search-preview-completion": 0.0006,
    "gpt-4o-mini-search-preview-2025-03-11-completion": 0.0006,
    # GPT-4o-search-preview input
    "gpt-4o-search-preview": 0.0025,
    "gpt-4o-search-preview-2025-03-11": 0.0025,
    # GPT-4o-search-preview output
    "gpt-4o-search-preview-completion": 0.01,
    "gpt-4o-search-preview-2025-03-11-completion": 0.01,
    # Computer-use-preview input
    "computer-use-preview": 0.003,
    "computer-use-preview-2025-03-11": 0.003,
    # Computer-use-preview output
    "computer-use-preview-completion": 0.012,
    "computer-use-preview-2025-03-11-completion": 0.012,
    # GPT-4 input
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-0613": 0.06,
    "gpt-4-vision-preview": 0.01,
    "gpt-4-1106-preview": 0.01,
    "gpt-4-0125-preview": 0.01,
    "gpt-4-turbo-preview": 0.01,
    "gpt-4-turbo": 0.01,
    "gpt-4-turbo-2024-04-09": 0.01,
    # GPT-4 output
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-0613-completion": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-4-32k-0613-completion": 0.12,
    "gpt-4-vision-preview-completion": 0.03,
    "gpt-4-1106-preview-completion": 0.03,
    "gpt-4-0125-preview-completion": 0.03,
    "gpt-4-turbo-preview-completion": 0.03,
    "gpt-4-turbo-completion": 0.03,
    "gpt-4-turbo-2024-04-09-completion": 0.03,
    # GPT-3.5 input
    # gpt-3.5-turbo points at gpt-3.5-turbo-0613 until Feb 16, 2024.
    # Switches to gpt-3.5-turbo-0125 after.
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-0125": 0.0005,
    "gpt-3.5-turbo-0301": 0.0015,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-1106": 0.001,
    "gpt-3.5-turbo-instruct": 0.0015,
    "gpt-3.5-turbo-16k": 0.003,
    "gpt-3.5-turbo-16k-0613": 0.003,
    # GPT-3.5 output
    # gpt-3.5-turbo points at gpt-3.5-turbo-0613 until Feb 16, 2024.
    # Switches to gpt-3.5-turbo-0125 after.
    "gpt-3.5-turbo-completion": 0.002,
    "gpt-3.5-turbo-0125-completion": 0.0015,
    "gpt-3.5-turbo-0301-completion": 0.002,
    "gpt-3.5-turbo-0613-completion": 0.002,
    "gpt-3.5-turbo-1106-completion": 0.002,
    "gpt-3.5-turbo-instruct-completion": 0.002,
    "gpt-3.5-turbo-16k-completion": 0.004,
    "gpt-3.5-turbo-16k-0613-completion": 0.004,
    # Azure GPT-35 input
    "gpt-35-turbo": 0.0015,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0125": 0.0005,
    "gpt-35-turbo-0301": 0.002,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0613": 0.0015,
    "gpt-35-turbo-instruct": 0.0015,
    "gpt-35-turbo-16k": 0.003,
    "gpt-35-turbo-16k-0613": 0.003,
    # Azure GPT-35 output
    "gpt-35-turbo-completion": 0.002,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0125-completion": 0.0015,
    "gpt-35-turbo-0301-completion": 0.002,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0613-completion": 0.002,
    "gpt-35-turbo-instruct-completion": 0.002,
    "gpt-35-turbo-16k-completion": 0.004,
    "gpt-35-turbo-16k-0613-completion": 0.004,
    # Others
    "text-ada-001": 0.0004,
    "ada": 0.0004,
    "text-babbage-001": 0.0005,
    "babbage": 0.0005,
    "text-curie-001": 0.002,
    "curie": 0.002,
    "text-davinci-003": 0.02,
    "text-davinci-002": 0.02,
    "code-davinci-002": 0.02,
    # Fine Tuned input
    "babbage-002-finetuned": 0.0016,
    "davinci-002-finetuned": 0.012,
    "gpt-3.5-turbo-0613-finetuned": 0.003,
    "gpt-3.5-turbo-1106-finetuned": 0.003,
    "gpt-3.5-turbo-0125-finetuned": 0.003,
    "gpt-4o-mini-2024-07-18-finetuned": 0.0003,
    "gpt-4o-mini-2024-07-18-finetuned-cached": 0.00015,
    # Fine Tuned output
    "babbage-002-finetuned-completion": 0.0016,
    "davinci-002-finetuned-completion": 0.012,
    "gpt-3.5-turbo-0613-finetuned-completion": 0.006,
    "gpt-3.5-turbo-1106-finetuned-completion": 0.006,
    "gpt-3.5-turbo-0125-finetuned-completion": 0.006,
    "gpt-4o-mini-2024-07-18-finetuned-completion": 0.0012,
    # Azure Fine Tuned input
    "babbage-002-azure-finetuned": 0.0004,
    "davinci-002-azure-finetuned": 0.002,
    "gpt-35-turbo-0613-azure-finetuned": 0.0015,
    # Azure Fine Tuned output
    "babbage-002-azure-finetuned-completion": 0.0004,
    "davinci-002-azure-finetuned-completion": 0.002,
    "gpt-35-turbo-0613-azure-finetuned-completion": 0.002,
    # Legacy fine-tuned models
    "ada-finetuned-legacy": 0.0016,
    "babbage-finetuned-legacy": 0.0024,
    "curie-finetuned-legacy": 0.012,
    "davinci-finetuned-legacy": 0.12,
}


class TokenType(Enum):
    """Token type enum."""

    PROMPT = auto()
    PROMPT_CACHED = auto()
    COMPLETION = auto()


def standardize_model_name(
    model_name: str,
    is_completion: bool = False,
    *,
    token_type: TokenType = TokenType.PROMPT,
) -> str:
    """
    Standardize the model name to a format that can be used in the OpenAI API.

    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.
            Defaults to False. Deprecated in favor of ``token_type``.
        token_type: Token type. Defaults to ``TokenType.PROMPT``.

    Returns:
        Standardized model name.

    """
    if is_completion:
        warn_deprecated(
            since="0.3.13",
            message=(
                "is_completion is deprecated. Use token_type instead. Example:\n\n"
                "from langchain_community.callbacks.openai_info import TokenType\n\n"
                "standardize_model_name('gpt-4o', token_type=TokenType.COMPLETION)\n"
            ),
            removal="1.0",
        )
        token_type = TokenType.COMPLETION
    model_name = model_name.lower()
    if ".ft-" in model_name:
        model_name = model_name.split(".ft-")[0] + "-azure-finetuned"
    if ":ft-" in model_name:
        model_name = model_name.split(":")[0] + "-finetuned-legacy"
    if "ft:" in model_name:
        model_name = model_name.split(":")[1] + "-finetuned"
    if token_type == TokenType.COMPLETION and (
        model_name.startswith("gpt-4")
        or model_name.startswith("gpt-3.5")
        or model_name.startswith("gpt-35")
        or model_name.startswith("o1-")
        or model_name.startswith("o3-")
        or model_name.startswith("o4-")
        or ("finetuned" in model_name and "legacy" not in model_name)
    ):
        return model_name + "-completion"
    if (
        token_type == TokenType.PROMPT_CACHED
        and (
            model_name.startswith("gpt-4o")
            or model_name.startswith("gpt-4.1")
            or model_name.startswith("o1")
            or model_name.startswith("o3")
            or model_name.startswith("o4")
        )
        and not (model_name.startswith("gpt-4o-2024-05-13"))
    ):
        return model_name + "-cached"
    else:
        return model_name


def get_openai_token_cost_for_model(
    model_name: str,
    num_tokens: int,
    is_completion: bool = False,
    *,
    token_type: TokenType = TokenType.PROMPT,
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False. Deprecated in favor of ``token_type``.
        token_type: Token type. Defaults to ``TokenType.PROMPT``.

    Returns:
        Cost in USD.
    """
    if is_completion:
        warn_deprecated(
            since="0.3.13",
            message=(
                "is_completion is deprecated. Use token_type instead. Example:\n\n"
                "from langchain_community.callbacks.openai_info import TokenType\n\n"
                "get_openai_token_cost_for_model('gpt-4o', 10, token_type=TokenType.COMPLETION)\n"  # noqa: E501
            ),
            removal="1.0",
        )
        token_type = TokenType.COMPLETION
    model_name = standardize_model_name(model_name, token_type=token_type)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


class OpenAICallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks OpenAI info."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    prompt_tokens_cached: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
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
            f"\t\tReasoning Tokens: {self.reasoning_tokens}\n"
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
        reasoning_tokens = 0

        if usage_metadata:
            token_usage = {"total_tokens": usage_metadata["total_tokens"]}
            completion_tokens = usage_metadata["output_tokens"]
            prompt_tokens = usage_metadata["input_tokens"]
            if response_model_name := (response_metadata or {}).get("model_name"):
                model_name = standardize_model_name(response_model_name)
            elif response.llm_output is None:
                model_name = ""
            else:
                model_name = standardize_model_name(
                    response.llm_output.get("model_name", "")
                )
            if "cache_read" in usage_metadata.get("input_token_details", {}):
                prompt_tokens_cached = usage_metadata["input_token_details"][
                    "cache_read"
                ]
            if "reasoning" in usage_metadata.get("output_token_details", {}):
                reasoning_tokens = usage_metadata["output_token_details"]["reasoning"]
        else:
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
            model_name = standardize_model_name(
                response.llm_output.get("model_name", "")
            )

        if model_name in MODEL_COST_PER_1K_TOKENS:
            uncached_prompt_tokens = prompt_tokens - prompt_tokens_cached
            uncached_prompt_cost = get_openai_token_cost_for_model(
                model_name, uncached_prompt_tokens, token_type=TokenType.PROMPT
            )
            cached_prompt_cost = get_openai_token_cost_for_model(
                model_name, prompt_tokens_cached, token_type=TokenType.PROMPT_CACHED
            )
            prompt_cost = uncached_prompt_cost + cached_prompt_cost
            completion_cost = get_openai_token_cost_for_model(
                model_name, completion_tokens, token_type=TokenType.COMPLETION
            )
        else:
            completion_cost = 0
            prompt_cost = 0

        # update shared state behind lock
        with self._lock:
            self.total_cost += prompt_cost + completion_cost
            self.total_tokens += token_usage.get("total_tokens", 0)
            self.prompt_tokens += prompt_tokens
            self.prompt_tokens_cached += prompt_tokens_cached
            self.completion_tokens += completion_tokens
            self.reasoning_tokens += reasoning_tokens
            self.successful_requests += 1

    def __copy__(self) -> "OpenAICallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "OpenAICallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
