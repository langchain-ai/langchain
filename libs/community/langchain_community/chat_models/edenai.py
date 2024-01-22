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

    To use, you should have the environment variable ``EDENAI_API_KEY`` set
    with your API key, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatEdenAI
            from langchain_core.messages import HumanMessage

            chat = ChatEdenAI(
                model="openai",
                model="gpt-4",
                max_tokens=256,
                temperature=0.75)

            messages = [HumanMessage(content="hello")]
            chat.invoke(messages)
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
            "Authorization": f"Bearer {self.edenai_api_key.get_secret_value()}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload, stream=True)

        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        for chunk_response in response.iter_lines():
            chunk = json.loads(chunk_response.decode())
            yield ChatGenerationChunk(message=AIMessageChunk(content=chunk["text"]))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        url = f"{self.edenai_api_url}/text/chat/stream"
        headers = {
            "Authorization": f"Bearer {self.edenai_api_key.get_secret_value()}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status >= 500:
                    raise Exception(f"EdenAI Server: Error {response.status}")
                elif response.status >= 400:
                    raise ValueError(
                        f"EdenAI received an invalid payload: {response.text}"
                    )
                elif response.status != 200:
                    raise Exception(
                        f"EdenAI returned an unexpected response with status "
                        f"{response.status}: {response.text}"
                    )

                async for chunk_response in response.content:
                    chunk = json.loads(chunk_response.decode())
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=chunk["text"])
                    )

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
            "Authorization": f"Bearer {self.edenai_api_key.get_secret_value()}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        request = Requests(headers=headers)
        response = request.post(url=url, data=payload)

        if response.status_code >= 500:
            raise Exception(f"EdenAI Server: Error {response.status_code}")
        elif response.status_code >= 400:
            raise ValueError(f"EdenAI received an invalid payload: {response.text}")
        elif response.status_code != 200:
            raise Exception(
                f"EdenAI returned an unexpected response with status "
                f"{response.status_code}: {response.text}"
            )

        data = response.json()
        provider_response = data[self.provider]
        if provider_response.get("status") == "fail":
            err_msg = provider_response.get("error", {}).get("message")
            raise Exception(err_msg)

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=data[self.provider]["generated_text"])
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
            "Authorization": f"Bearer {self.edenai_api_key.get_secret_value()}",
            "User-Agent": self.get_user_agent(),
        }
        formatted_data = _format_edenai_messages(messages=messages)
        payload: Dict[str, Any] = {
            "providers": self.provider,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}

        if self.model is not None:
            payload["settings"] = {self.provider: self.model}

        async with ClientSession() as session:
            async with session.post(url, json=payload, headers=headers) as response:
                if response.status >= 500:
                    raise Exception(f"EdenAI Server: Error {response.status}")
                elif response.status >= 400:
                    raise ValueError(
                        f"EdenAI received an invalid payload: {response.text}"
                    )
                elif response.status != 200:
                    raise Exception(
                        f"EdenAI returned an unexpected response with status "
                        f"{response.status}: {response.text}"
                    )
                data = await response.json()
                provider_response = data[self.provider]
                if provider_response.get("status") == "fail":
                    err_msg = provider_response.get("error", {}).get("message")
                    raise Exception(err_msg)

                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=data[self.provider]["generated_text"]
                            )
                        )
                    ],
                    llm_output=data,
                )
