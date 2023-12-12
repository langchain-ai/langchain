from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    cast,
)

import anthropic
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseLanguageModel
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.prompt_values import PromptValue
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    check_package_version,
    get_from_dict_or_env,
    get_pydantic_field_names,
)
from langchain_core.utils.utils import build_extra_kwargs, convert_to_secret_str


class _AnthropicCommon(BaseLanguageModel):
    client: Any = None  #: :meta private:
    async_client: Any = None  #: :meta private:
    model: str = Field(default="claude-2", alias="model_name")
    """Model name to use."""

    max_tokens_to_sample: int = Field(default=256, alias="max_tokens")
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    streaming: bool = False
    """Whether to stream the results."""

    default_request_timeout: Optional[float] = None
    """Timeout for requests to Anthropic Completion API. Default is 600 seconds."""

    anthropic_api_url: Optional[str] = None

    anthropic_api_key: Optional[SecretStr] = None

    HUMAN_PROMPT: Optional[str] = None
    AI_PROMPT: Optional[str] = None
    count_tokens: Optional[Callable[[str], int]] = None
    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    @root_validator(pre=True)
    def build_extra(cls, values: Dict) -> Dict:
        extra = values.get("model_kwargs", {})
        all_required_field_names = get_pydantic_field_names(cls)
        values["model_kwargs"] = build_extra_kwargs(
            extra, values, all_required_field_names
        )
        return values

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["anthropic_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "anthropic_api_key", "ANTHROPIC_API_KEY")
        )
        # Get custom api url from environment.
        values["anthropic_api_url"] = get_from_dict_or_env(
            values,
            "anthropic_api_url",
            "ANTHROPIC_API_URL",
            default="https://api.anthropic.com",
        )

        check_package_version("anthropic", gte_version="0.3")
        values["client"] = anthropic.Anthropic(
            base_url=values["anthropic_api_url"],
            api_key=values["anthropic_api_key"].get_secret_value(),
            timeout=values["default_request_timeout"],
        )
        values["async_client"] = anthropic.AsyncAnthropic(
            base_url=values["anthropic_api_url"],
            api_key=values["anthropic_api_key"].get_secret_value(),
            timeout=values["default_request_timeout"],
        )
        values["HUMAN_PROMPT"] = anthropic.HUMAN_PROMPT
        values["AI_PROMPT"] = anthropic.AI_PROMPT
        values["count_tokens"] = values["client"].count_tokens

        return values

    @property
    def _default_params(self) -> Mapping[str, Any]:
        """Get the default parameters for calling Anthropic API."""
        d = {
            "max_tokens_to_sample": self.max_tokens_to_sample,
            "model": self.model,
        }
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.top_k is not None:
            d["top_k"] = self.top_k
        if self.top_p is not None:
            d["top_p"] = self.top_p
        return {**d, **self.model_kwargs}

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {**{}, **self._default_params}

    def _get_anthropic_stop(self, stop: Optional[List[str]] = None) -> List[str]:
        if not self.HUMAN_PROMPT or not self.AI_PROMPT:
            raise NameError("Please ensure the anthropic package is loaded")

        if stop is None:
            stop = []

        # Never want model to invent new turns of Human / Assistant dialog.
        stop.extend([self.HUMAN_PROMPT])

        return stop


def _convert_one_message_to_text(
    message: BaseMessage,
    human_prompt: str,
    ai_prompt: str,
) -> str:
    content = cast(str, message.content)
    if isinstance(message, ChatMessage):
        message_text = f"\n\n{message.role.capitalize()}: {content}"
    elif isinstance(message, HumanMessage):
        message_text = f"{human_prompt} {content}"
    elif isinstance(message, AIMessage):
        message_text = f"{ai_prompt} {content}"
    elif isinstance(message, SystemMessage):
        message_text = content
    else:
        raise ValueError(f"Got unknown type {message}")
    return message_text


def convert_messages_to_prompt_anthropic(
    messages: List[BaseMessage],
    *,
    human_prompt: str = "\n\nHuman:",
    ai_prompt: str = "\n\nAssistant:",
) -> str:
    """Format a list of messages into a full prompt for the Anthropic model
    Args:
        messages (List[BaseMessage]): List of BaseMessage to combine.
        human_prompt (str, optional): Human prompt tag. Defaults to "\n\nHuman:".
        ai_prompt (str, optional): AI prompt tag. Defaults to "\n\nAssistant:".
    Returns:
        str: Combined string with necessary human_prompt and ai_prompt tags.
    """

    messages = messages.copy()  # don't mutate the original list
    if not isinstance(messages[-1], AIMessage):
        messages.append(AIMessage(content=""))

    text = "".join(
        _convert_one_message_to_text(message, human_prompt, ai_prompt)
        for message in messages
    )

    # trim off the trailing ' ' that might come from the "Assistant: "
    return text.rstrip()


class ChatAnthropic(BaseChatModel, _AnthropicCommon):
    """`Anthropic` chat large language models.

    To use, you should have the
    environment variable ``ANTHROPIC_API_KEY`` set with your API key, or pass
    it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic
            model = ChatAnthropic(model="<model_name>", anthropic_api_key="my-api-key")
    """

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"anthropic_api_key": "ANTHROPIC_API_KEY"}

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "chat_models", "anthropic"]

    def _convert_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """Format a list of messages into a full prompt for the Anthropic model
        Args:
            messages (List[BaseMessage]): List of BaseMessage to combine.
        Returns:
            str: Combined string with necessary HUMAN_PROMPT and AI_PROMPT tags.
        """
        prompt_params = {}
        if self.HUMAN_PROMPT:
            prompt_params["human_prompt"] = self.HUMAN_PROMPT
        if self.AI_PROMPT:
            prompt_params["ai_prompt"] = self.AI_PROMPT
        return convert_messages_to_prompt_anthropic(messages=messages, **prompt_params)

    def convert_prompt(self, prompt: PromptValue) -> str:
        return self._convert_messages_to_prompt(prompt.to_messages())

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params, **kwargs}
        if stop:
            params["stop_sequences"] = stop

        stream_resp = self.client.completions.create(**params, stream=True)
        for data in stream_resp:
            delta = data.completion
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                run_manager.on_llm_new_token(delta)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        prompt = self._convert_messages_to_prompt(messages)
        params: Dict[str, Any] = {"prompt": prompt, **self._default_params, **kwargs}
        if stop:
            params["stop_sequences"] = stop

        stream_resp = await self.async_client.completions.create(**params, stream=True)
        async for data in stream_resp:
            delta = data.completion
            yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
            if run_manager:
                await run_manager.on_llm_new_token(delta)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)
        prompt = self._convert_messages_to_prompt(
            messages,
        )
        params: Dict[str, Any] = {
            "prompt": prompt,
            **self._default_params,
            **kwargs,
        }
        if stop:
            params["stop_sequences"] = stop
        response = self.client.completions.create(**params)
        completion = response.completion
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)
        prompt = self._convert_messages_to_prompt(
            messages,
        )
        params: Dict[str, Any] = {
            "prompt": prompt,
            **self._default_params,
            **kwargs,
        }
        if stop:
            params["stop_sequences"] = stop
        response = await self.async_client.completions.create(**params)
        completion = response.completion
        message = AIMessage(content=completion)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        if not self.count_tokens:
            raise NameError("Please ensure the anthropic package is loaded")
        return self.count_tokens(text)
