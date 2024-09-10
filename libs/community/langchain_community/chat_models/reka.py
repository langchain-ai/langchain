from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
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
from langchain_core.pydantic_v1 import Field, PrivateAttr
from langchain_core.utils import get_from_dict_or_env

try:
    from reka.client import AsyncReka, Reka
except ImportError:
    raise ValueError(
        "Reka is not installed. Please install it with `pip install reka-api`."
    )

REKA_MODELS = [
    "reka-edge",
    "reka-flash",
    "reka-core",
    "reka-core-20240501",
]

DEFAULT_REKA_MODEL = "reka-core-20240501"

def get_role(message: BaseMessage) -> str:
    """Get the role of the message."""
    if isinstance(message, (ChatMessage, HumanMessage)):
        return "user"
    elif isinstance(message, AIMessage):
        return "assistant"
    elif isinstance(message, SystemMessage):
        return "system"
    else:
        raise ValueError(f"Got unknown type {message}")

def process_messages_for_reka(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Process messages for Reka format."""
    reka_messages = []
    system_message = None

    for message in messages:
        if isinstance(message, SystemMessage):
            if system_message is None:
                system_message = message.content
            else:
                raise ValueError("Multiple system messages are not supported.")
        else:
            content = message.content
            if system_message and isinstance(message, HumanMessage):
                content = f"{system_message}\n{content}"
                system_message = None
            reka_messages.append({"role": get_role(message), "content": content})

    return reka_messages

class ChatReka(BaseChatModel):
    """Reka chat large language models."""

    model: str = Field(
        default=DEFAULT_REKA_MODEL, 
        description="The Reka model to use."
    )
    temperature: float = Field(
        default=0.7, 
        description="The sampling temperature."
    )
    max_tokens: int = Field(
        default=512, 
        description="The maximum number of tokens to generate."
    )
    api_key: str = Field(
        default=None, 
        description="The API key for Reka."
    )
    streaming: bool = Field(
        default=False, 
        description="Whether to stream the response."
    )

    _client: Reka = PrivateAttr()
    _aclient: AsyncReka = PrivateAttr()

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        api_key = get_from_dict_or_env(kwargs, "api_key", "REKA_API_KEY")
        self._client = Reka(api_key=api_key)
        self._aclient = AsyncReka(api_key=api_key)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "reka-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Reka API."""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reka_messages = process_messages_for_reka(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop

        stream = self._client.chat.create_stream(messages=reka_messages, **params)

        for chunk in stream:
            content = chunk.responses[0].chunk.content
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(content, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        reka_messages = process_messages_for_reka(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop

        stream = await self._aclient.chat.create_stream(
            messages=reka_messages, 
            **params
        )

        async for chunk in stream:
            content = chunk.responses[0].chunk.content
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=content))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(content, chunk=chunk)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return generate_from_stream(
                self._stream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )

        reka_messages = process_messages_for_reka(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop
        response = self._client.chat.create(messages=reka_messages, **params)

        message = AIMessage(content=response.responses[0].message.content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            return await agenerate_from_stream(
                self._astream(messages, stop=stop, run_manager=run_manager, **kwargs)
            )

        reka_messages = process_messages_for_reka(messages)
        params = {**self._default_params, **kwargs}
        if stop:
            params["stop"] = stop
        response = await self._aclient.chat.create(messages=reka_messages, **params)

        message = AIMessage(content=response.responses[0].message.content)
        return ChatResult(generations=[ChatGeneration(message=message)])

    def get_num_tokens(self, text: str) -> int:
        """Calculate number of tokens."""
        raise NotImplementedError("Token counting is not implemented for Reka models.")
