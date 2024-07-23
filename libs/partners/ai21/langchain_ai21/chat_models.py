import asyncio
from functools import partial
from typing import Any, Dict, Iterator, List, Mapping, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    LangSmithParams,
    generate_from_stream,
)
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import root_validator

from langchain_ai21.ai21_base import AI21Base
from langchain_ai21.chat.chat_adapter import ChatAdapter
from langchain_ai21.chat.chat_factory import create_chat_adapter


class ChatAI21(BaseChatModel, AI21Base):
    """ChatAI21 chat model. Different model types support different parameters and
    different parameter values. Please read the [AI21 reference documentation]
    (https://docs.ai21.com/reference) for your model to understand which parameters
    are available.

    Example:
        .. code-block:: python

            from langchain_ai21 import ChatAI21


            model = ChatAI21(
                # defaults to os.environ.get("AI21_API_KEY")
                api_key="my_api_key"
            )
    """

    model: str
    """Model type you wish to interact with. 
        You can view the options at https://github.com/AI21Labs/ai21-python?tab=readme-ov-file#model-types"""
    num_results: int = 1
    """The number of responses to generate for a given prompt."""
    stop: Optional[List[str]] = None
    """Default stop sequences."""

    max_tokens: int = 16
    """The maximum number of tokens to generate for each response."""

    min_tokens: int = 0
    """The minimum number of tokens to generate for each response.
    _Not supported for all models._"""

    temperature: float = 0.7
    """A value controlling the "creativity" of the model's responses."""

    top_p: float = 1
    """A value controlling the diversity of the model's responses."""

    top_k_return: int = 0
    """The number of top-scoring tokens to consider for each generation step.
    _Not supported for all models._"""

    frequency_penalty: Optional[Any] = None
    """A penalty applied to tokens that are frequently generated.
    _Not supported for all models._"""

    presence_penalty: Optional[Any] = None
    """ A penalty applied to tokens that are already present in the prompt.
    _Not supported for all models._"""

    count_penalty: Optional[Any] = None
    """A penalty applied to tokens based on their frequency 
    in the generated responses. _Not supported for all models._"""

    n: int = 1
    """Number of chat completions to generate for each prompt."""
    streaming: bool = False

    _chat_adapter: ChatAdapter

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values = super().validate_environment(values)
        model = values.get("model")

        values["_chat_adapter"] = create_chat_adapter(model)  # type: ignore

        return values

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ai21"

    @property
    def _default_params(self) -> Mapping[str, Any]:
        base_params = {
            "model": self.model,
            "num_results": self.num_results,
            "max_tokens": self.max_tokens,
            "min_tokens": self.min_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k_return": self.top_k_return,
            "n": self.n,
        }
        if self.stop:
            base_params["stop_sequences"] = self.stop

        if self.count_penalty is not None:
            base_params["count_penalty"] = self.count_penalty.to_dict()

        if self.frequency_penalty is not None:
            base_params["frequency_penalty"] = self.frequency_penalty.to_dict()

        if self.presence_penalty is not None:
            base_params["presence_penalty"] = self.presence_penalty.to_dict()

        return base_params

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get standard params for tracing."""
        params = self._get_invocation_params(stop=stop, **kwargs)
        ls_params = LangSmithParams(
            ls_provider="ai21",
            ls_model_name=self.model,
            ls_model_type="chat",
            ls_temperature=params.get("temperature", self.temperature),
        )
        if ls_max_tokens := params.get("max_tokens", self.max_tokens):
            ls_params["ls_max_tokens"] = ls_max_tokens
        if ls_stop := stop or params.get("stop", None) or self.stop:
            ls_params["ls_stop"] = ls_stop
        return ls_params

    def _build_params_for_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Mapping[str, Any]:
        params = {}
        converted_messages = self._chat_adapter.convert_messages(messages)

        if stop is not None:
            if "stop" in kwargs:
                raise ValueError("stop is defined in both stop and kwargs")
            params["stop_sequences"] = stop

        return {
            **converted_messages,
            **self._default_params,
            **params,
            **kwargs,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream or self.streaming

        if should_stream:
            return self._handle_stream_from_generate(
                messages=messages,
                stop=stop,
                run_manager=run_manager,
                **kwargs,
            )

        params = self._build_params_for_request(
            messages=messages,
            stop=stop,
            stream=should_stream,
            **kwargs,
        )

        messages = self._chat_adapter.call(self.client, **params)
        generations = [ChatGeneration(message=message) for message in messages]

        return ChatResult(generations=generations)

    def _handle_stream_from_generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        stream_iter = self._stream(
            messages=messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )
        return generate_from_stream(stream_iter)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._build_params_for_request(
            messages=messages,
            stop=stop,
            stream=True,
            **kwargs,
        )

        for chunk in self._chat_adapter.call(self.client, **params):
            if run_manager and isinstance(chunk.message.content, str):
                run_manager.on_llm_new_token(token=chunk.message.content, chunk=chunk)
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await asyncio.get_running_loop().run_in_executor(
            None, partial(self._generate, **kwargs), messages, stop, run_manager
        )

    def _get_system_message_from_message(self, message: BaseMessage) -> str:
        if not isinstance(message.content, str):
            raise ValueError(
                f"System Message must be of type str. Got {type(message.content)}"
            )

        return message.content
