"""Callback Handler that collects token usage from OpenAI models."""
import warnings
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union
from uuid import UUID

from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage, LLMResult
from langchain.utils.openai import convert_message_to_dict

if TYPE_CHECKING:
    import tiktoken

MODEL_COST_PER_1K_TOKENS = {
    # GPT-4 input
    "gpt-4": 0.03,
    "gpt-4-0314": 0.03,
    "gpt-4-0613": 0.03,
    "gpt-4-32k": 0.06,
    "gpt-4-32k-0314": 0.06,
    "gpt-4-32k-0613": 0.06,
    # GPT-4 output
    "gpt-4-completion": 0.06,
    "gpt-4-0314-completion": 0.06,
    "gpt-4-0613-completion": 0.06,
    "gpt-4-32k-completion": 0.12,
    "gpt-4-32k-0314-completion": 0.12,
    "gpt-4-32k-0613-completion": 0.12,
    # GPT-3.5 input
    "gpt-3.5-turbo": 0.0015,
    "gpt-3.5-turbo-0301": 0.0015,
    "gpt-3.5-turbo-0613": 0.0015,
    "gpt-3.5-turbo-instruct": 0.0015,
    "gpt-3.5-turbo-16k": 0.003,
    "gpt-3.5-turbo-16k-0613": 0.003,
    # GPT-3.5 output
    "gpt-3.5-turbo-completion": 0.002,
    "gpt-3.5-turbo-0301-completion": 0.002,
    "gpt-3.5-turbo-0613-completion": 0.002,
    "gpt-3.5-turbo-instruct-completion": 0.002,
    "gpt-3.5-turbo-16k-completion": 0.004,
    "gpt-3.5-turbo-16k-0613-completion": 0.004,
    # Azure GPT-35 input
    "gpt-35-turbo": 0.0015,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0301": 0.0015,  # Azure OpenAI version of ChatGPT
    "gpt-35-turbo-0613": 0.0015,
    "gpt-35-turbo-instruct": 0.0015,
    "gpt-35-turbo-16k": 0.003,
    "gpt-35-turbo-16k-0613": 0.003,
    # Azure GPT-35 output
    "gpt-35-turbo-completion": 0.002,  # Azure OpenAI version of ChatGPT
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
    "ada-finetuned": 0.0016,
    "babbage-finetuned": 0.0024,
    "curie-finetuned": 0.012,
    "davinci-finetuned": 0.12,
}


def standardize_model_name(
    model_name: str,
    is_completion: bool = False,
) -> str:
    """
    Standardize the model name to a format that can be used in the OpenAI API.

    Args:
        model_name: Model name to standardize.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Standardized model name.

    """
    model_name = model_name.lower()
    if "ft-" in model_name:
        return model_name.split(":")[0] + "-finetuned"
    elif is_completion and (
        model_name.startswith("gpt-4")
        or model_name.startswith("gpt-3.5")
        or model_name.startswith("gpt-35")
    ):
        return model_name + "-completion"
    else:
        return model_name


def get_openai_token_cost_for_model(
    model_name: str, num_tokens: int, is_completion: bool = False
) -> float:
    """
    Get the cost in USD for a given model and number of tokens.

    Args:
        model_name: Name of the model
        num_tokens: Number of tokens.
        is_completion: Whether the model is used for completion or not.
            Defaults to False.

    Returns:
        Cost in USD.
    """
    model_name = standardize_model_name(model_name, is_completion=is_completion)
    if model_name not in MODEL_COST_PER_1K_TOKENS:
        raise ValueError(
            f"Unknown model: {model_name}. Please provide a valid OpenAI model name."
            "Known models are: " + ", ".join(MODEL_COST_PER_1K_TOKENS.keys())
        )
    return MODEL_COST_PER_1K_TOKENS[model_name] * (num_tokens / 1000)


def _import_tiktoken() -> Union["tiktoken", None]:
    try:
        import tiktoken
    except ImportError:
        warnings.warn(
            "tiktoken is not installed. "
            "Streaming functionality of OpenAICallbackHandler will not work."
            "Please install tiktoken to enable streaming support."
        )
        return None
    return tiktoken


class OpenAICallbackHandler(BaseCallbackHandler):
    """Callback Handler that tracks OpenAI info."""

    def __init__(self) -> None:
        super().__init__()
        self.current_model_name: Dict[UUID, str] = {}
        self.token_count: Dict[str, Dict[UUID, Dict[str, Any]]] = {}
        self.tokens: Dict[UUID, List[str]] = {}
        self.tiktoken_ = _import_tiktoken()

    @property
    def total_tokens(self) -> int:
        if len(self.token_count) > 1:
            warnings.warn(
                "You are using multiple models. "
                "The token count will be inaccurate. "
                "Please use get_total_tokens_for_model "
                "to get the token count for a specific model name."
            )
        total_tokens = 0
        for model_name, token_count in self.token_count.items():
            for run_id in token_count:
                total_tokens += token_count[run_id]["prompt_tokens"]
                total_tokens += token_count[run_id]["completion_tokens"]
        return total_tokens

    @property
    def prompt_tokens(self) -> int:
        if len(self.token_count) > 1:
            warnings.warn(
                "You are using multiple models. "
                "The token count will be inaccurate. "
                "Please use get_prompt_tokens_for_model "
                "to get the token count for a specific model name."
            )
        prompt_tokens = 0
        for model_name in self.token_count.keys():
            prompt_tokens += self.get_prompt_tokens_for_model(model_name)
        return prompt_tokens

    @property
    def completion_tokens(self) -> int:
        if len(self.token_count) > 1:
            warnings.warn(
                "You are using multiple models. "
                "The token count will be inaccurate. "
                "Please use get_completion_tokens_for_model "
                "to get the token count for a specific model name."
            )
        completion_tokens = 0
        for model_name in self.token_count.keys():
            completion_tokens += self.get_completion_tokens_for_model(model_name)
        return completion_tokens

    @property
    def successful_requests(self) -> int:
        successful_requests = 0
        for model_name in self.token_count.keys():
            for run_id in self.token_count[model_name]:
                successful_requests += self.token_count[model_name][run_id][
                    "successful_requests"
                ]
        return successful_requests

    @property
    def total_cost(self) -> float:
        total_cost = 0.0
        for model_name in self.token_count.keys():
            if model_name in MODEL_COST_PER_1K_TOKENS:
                total_cost += get_openai_token_cost_for_model(
                    model_name,
                    self.get_prompt_tokens_for_model(model_name),
                    is_completion=False,
                )
                total_cost += get_openai_token_cost_for_model(
                    model_name,
                    self.get_completion_tokens_for_model(model_name),
                    is_completion=True,
                )
        return total_cost

    @property
    def always_verbose(self) -> bool:
        """Whether to call verbose callbacks even if verbose is False."""
        return True

    def get_prompt_tokens_for_model(self, model_name: str) -> int:
        """Get the number of prompt tokens for a given model."""
        model_name = standardize_model_name(model_name)
        prompt_tokens = 0
        for run_id in self.token_count[model_name]:
            prompt_tokens += self.token_count[model_name][run_id]["prompt_tokens"]
        return prompt_tokens

    def get_completion_tokens_for_model(self, model_name: str) -> int:
        """Get the number of completion tokens for a given model."""
        model_name = standardize_model_name(model_name)
        completion_tokens = 0
        for run_id in self.token_count[model_name]:
            completion_tokens += self.token_count[model_name][run_id][
                "completion_tokens"
            ]
        return completion_tokens

    def get_total_tokens_for_model(self, model_name: str) -> int:
        """Get the total number of tokens for a given model."""
        return self.get_prompt_tokens_for_model(
            model_name
        ) + self.get_completion_tokens_for_model(model_name)

    def get_total_cost_for_model(self, model_name: str) -> float:
        """Get the total cost in USD for a given model."""
        prompt_tokens = self.get_prompt_tokens_for_model(model_name)
        completion_tokens = self.get_completion_tokens_for_model(model_name)
        return get_openai_token_cost_for_model(
            model_name, prompt_tokens, is_completion=False
        ) + get_openai_token_cost_for_model(
            model_name, completion_tokens, is_completion=True
        )

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> None:
        is_streaming = serialized.get("kwargs", {}).get("streaming", False)
        if is_streaming:
            model_name = standardize_model_name(
                kwargs.get("invocation_params", {}).get("model_name")
            )
            self.current_model_name[run_id] = model_name
            self._prepare_model_run(model_name, run_id)
            self.token_count[model_name][run_id][
                "prompt_tokens"
            ] += self._get_num_tokens_from_prompts(prompts, run_id)

    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        **kwargs: Any,
    ) -> Any:
        is_streaming = serialized.get("kwargs", {}).get("streaming", False)
        if is_streaming:
            model_name = standardize_model_name(
                kwargs.get("invocation_params", {}).get("model_name")
            )
            self.current_model_name[run_id] = model_name
            self._prepare_model_run(model_name, run_id)
            for message in messages:
                self.token_count[model_name][run_id][
                    "prompt_tokens"
                ] += self._get_num_tokens_from_messages(message, run_id)

    def on_llm_new_token(self, token: str, *, run_id: UUID, **kwargs: Any) -> None:
        if (
            self.current_model_name[run_id]
            and self.current_model_name[run_id] in MODEL_COST_PER_1K_TOKENS
        ):
            self._prepare_model_run(self.current_model_name[run_id], run_id)
            if not self.tokens.get(run_id):
                self.tokens[run_id] = []
            self.tokens[run_id].append(token)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        """Collect token usage."""
        if response.llm_output:
            model_name = standardize_model_name(
                response.llm_output.get("model_name", "")
            )
            if not model_name:
                return None
            self._prepare_model_run(model_name, run_id)
            if model_name in MODEL_COST_PER_1K_TOKENS:
                token_usage = response.llm_output.get("token_usage", {}).get(
                    "completion_tokens", 0
                )
                prompt_token_usage = response.llm_output.get("token_usage", {}).get(
                    "prompt_tokens", 0
                )
                self.token_count[model_name][run_id]["completion_tokens"] += token_usage
                if self.token_count[model_name][run_id]["prompt_tokens"] == 0:
                    self.token_count[model_name][run_id][
                        "prompt_tokens"
                    ] = prompt_token_usage
            self.token_count[model_name][run_id]["successful_requests"] += 1
        if self.tokens.get(run_id):
            completion_tokens = len(self.tokens[run_id])
            model_name = self.current_model_name[run_id]
            if completion_tokens > 0:
                # OpenAI will send empty first and last tokens for completion.
                if self.tokens[run_id][0] == "":
                    completion_tokens -= 1
                if self.tokens[run_id][-1] == "":
                    completion_tokens -= 1
                self.token_count[model_name][run_id][
                    "completion_tokens"
                ] += completion_tokens
                del self.tokens[run_id]
        if run_id in self.current_model_name:
            del self.current_model_name[run_id]

    def _prepare_model_run(self, model_name: Union[str, None], run_id: UUID) -> None:
        if model_name is None:
            return
        if model_name not in self.token_count:
            self.token_count[model_name] = {}
        if run_id not in self.token_count[model_name]:
            self.token_count[model_name][run_id] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "successful_requests": 0,
            }

    def _get_encoding_model(self, run_id: UUID) -> Tuple[str, "tiktoken.Encoding"]:
        if not self.tiktoken_:
            raise ImportError("tiktoken is not installed.")
        model = self.current_model_name[run_id]
        if model is None:
            raise ValueError("Model name should be set.")
        if model.startswith("gpt-35-turbo"):
            model = "gpt-3.5-turbo"
        try:
            encoding = self.tiktoken_.encoding_for_model(model)
        except KeyError:
            warnings.warn("Warning: model not found. Using cl100k_base encoding.")
            model = "cl100k_base"
            encoding = self.tiktoken_.get_encoding(model)
        return model, encoding

    def _get_num_tokens_from_messages(
        self, messages: List[BaseMessage], run_id: UUID
    ) -> int:
        try:
            model, encoding = self._get_encoding_model(run_id)
        except ImportError:
            return 0
        if model.startswith("gpt-3.5-turbo-0301"):
            tokens_per_message = 4
            tokens_per_name = -1
        elif model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            tokens_per_message = 3
            tokens_per_name = 1
        num_tokens = 0
        messages_dict = [convert_message_to_dict(m) for m in messages]
        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(str(value)))
                if key == "name":
                    num_tokens += tokens_per_name
        num_tokens += 3
        return num_tokens

    def _get_num_tokens_from_prompts(self, prompts: List[str], run_id: UUID) -> int:
        try:
            model, encoding = self._get_encoding_model(run_id)
        except ImportError:
            return 0
        num_tokens = 0
        for prompt in prompts:
            num_tokens += len(encoding.encode(prompt))
        return num_tokens

    def __repr__(self) -> str:
        result = "OpenAI Token Usage:\n"
        for model_name in self.token_count:
            result += f"{model_name}:\n"
            result += (
                f"\tPrompt tokens: {self.get_prompt_tokens_for_model(model_name)}\n"
            )
            result += (
                f"\tCompletion tokens: "
                f"{self.get_completion_tokens_for_model(model_name)}\n"
            )
        result += f"Total cost (USD): ${self.total_cost}"
        return result

    def __copy__(self) -> "OpenAICallbackHandler":
        """Return a copy of the callback handler."""
        return self

    def __deepcopy__(self, memo: Any) -> "OpenAICallbackHandler":
        """Return a deep copy of the callback handler."""
        return self
