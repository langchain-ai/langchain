import json
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from aiohttp import ClientSession
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
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Extra, Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

from langchain_community.utilities.requests import Requests


def _message_role(type: str) -> str:
    role_mapping = {"ai": "assistant", "human": "user", "chat": "user"}

    if type in role_mapping:
        return role_mapping[type]
    else:
        raise ValueError(f"Unknown type: {type}")


def _format_edenai_messages(messages: List[BaseMessage]) -> Dict[str, Any]:
    system = None
    formatted_messages = []
    text = messages[-1].content
    for i, message in enumerate(messages[:-1]):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            system = message.content
        else:
            formatted_messages.append(
                {
                    "role": _message_role(message.type),
                    "message": message.content,
                }
            )
    return {
        "text": text,
        "previous_history": formatted_messages,
        "chatbot_global_action": system,
    }


class ChatEdenAI(BaseChatModel):
    """`EdenAI` chat large language models.

    `EdenAI` is a versatile platform that allows you to access various language models
    from different providers such as Google, OpenAI, Cohere, Mistral and more.

    To get started, make sure you have the environment variable ``EDENAI_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.

    Additionally, `EdenAI` provides the flexibility to choose from a variety of models,
    including the ones like "gpt-4".

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatEdenAI
            from langchain_core.messages import HumanMessage

            # Initialize `ChatEdenAI` with the desired configuration
            chat = ChatEdenAI(
                provider="openai",
                model="gpt-4",
                max_tokens=256,
                temperature=0.75)

            # Create a list of messages to interact with the model
            messages = [HumanMessage(content="hello")]

            # Invoke the model with the provided messages
            chat.invoke(messages)

    `EdenAI` goes beyond mere model invocation. It empowers you with advanced features :

    - **Multiple Providers**: access to a diverse range of llms offered by various
     providers giving you the freedom to choose the best-suited model for your use case.

    - **Fallback Mechanism**: Set a fallback mechanism to ensure seamless operations
        even if the primary provider is unavailable, you can easily switches to an
        alternative provider.

    - **Usage Statistics**: Track usage statistics on a per-project
    and per-API key basis.
    This feature allows you to monitor and manage resource consumption effectively.

    - **Monitoring and Observability**: `EdenAI` provides comprehensive monitoring
    and observability tools on the platform.

    Example of setting up a fallback mechanism:
        .. code-block:: python

            # Initialize `ChatEdenAI` with a fallback provider
            chat_with_fallback = ChatEdenAI(
                provider="openai",
                model="gpt-4",
                max_tokens=256,
                temperature=0.75,
                fallback_provider="google")

    you can find more details here : https://docs.edenai.co/reference/text_chat_create
    """

    provider: str = "openai"
    """chat provider to use (eg: openai,google etc.)"""

    model: Optional[str] = None
    """
    model name for above provider (eg: 'gpt-4' for openai)
    available models are shown on https://docs.edenai.co/ under 'available providers'
    """

    max_tokens: int = 256
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = 0
    """A non-negative float that tunes the degree of randomness in generation."""

    streaming: bool = False
    """Whether to stream the results."""

    fallback_providers: Optional[str] = None
    """Providers in this will be used as fallback if the call to provider fails."""

    edenai_api_url: str = "https://api.edenai.run/v2"

    edenai_api_key: Optional[SecretStr] = Field(None, description="EdenAI API Token")

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["edenai_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "edenai_api_key", "EDENAI_API_KEY")
        )
        return values

    @staticmethod
    def get_user_agent() -> str:
        from langchain_community import __version__

        return f"langchain/{__version__}"

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "edenai-chat"

    @property
    def _api_key(self) -> str:
        if self.edenai_api_key:
            return self.edenai_api_key.get_secret_value()
        return ""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Call out to EdenAI's chat endpoint."""
        url = f"{self.edenai_api_url}/text/chat/stream"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload, stream=True)
        response.raise_for_status()

        for chunk_response in response.iter_lines():
            chunk = json.loads(chunk_response.decode())
            token = chunk["text"]
            cg_chunk = ChatGenerationChunk(message=AIMessageChunk(content=token))
            if run_manager:
                run_manager.on_llm_new_token(token, chunk=cg_chunk)
            yield cg_chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        url = f"{self.edenai_api_url}/text/chat/stream"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                async for chunk_response in response.content:
                    chunk = json.loads(chunk_response.decode())
                    token = chunk["text"]
                    cg_chunk = ChatGenerationChunk(
                        message=AIMessageChunk(content=token)
                    )
                    if run_manager:
                        await run_manager.on_llm_new_token(
                            token=chunk["text"], chunk=cg_chunk
                        )
                    yield cg_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to EdenAI's chat endpoint."""
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        url = f"{self.edenai_api_url}/text/chat"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload)

        response.raise_for_status()
        data = response.json()
        provider_response = data[self.provider]

        if self.fallback_providers:
            fallback_response = data.get(self.fallback_providers)
            if fallback_response:
                provider_response = fallback_response

        if provider_response.get("status") == "fail":
            err_msg = provider_response.get("error", {}).get("message")
            raise Exception(err_msg)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=provider_response["generated_text"])
                )
            ],
            llm_output=data,
        )

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

        url = f"{self.edenai_api_url}/text/chat"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                provider_response = data[self.provider]

                if self.fallback_providers:
                    fallback_response = data.get(self.fallback_providers)
                    if fallback_response:
                        provider_response = fallback_response

                if provider_response.get("status") == "fail":
                    err_msg = provider_response.get("error", {}).get("message")
                    raise Exception(err_msg)

                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=provider_response["generated_text"]
                            )
                        )
                    ],
                    llm_output=data,
                )
