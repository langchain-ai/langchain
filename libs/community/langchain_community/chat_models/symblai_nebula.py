import json
import requests
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
    ChatMessage,
    HumanMessage,
    SystemMessage,
)

from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

def _message_role(type: str) -> str:
    role_mapping = {"ai": "assistant", "human": "user", "chat": "user"}

    if type in role_mapping:
        return role_mapping[type]
    else:
        raise ValueError(f"Unknown type: {type}")


def _format_nebula_messages(messages: List[BaseMessage]) -> Dict[str, Any]:
    # print("message: ", messages)
    system = None
    formatted_messages = []
    text = messages[-1].content
    # print(text)
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
        "system_prompt": system,
        "messages" : [
            {
                "role": "human",
                "text": text
            }
        ]
    }


class ChatNebula(BaseChatModel):
    """`Nebula` chat large language models.

    To use, you should have the environment variable ``NEBULA_API_KEY`` set with your API key, 
    or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_community.chat_models import ChatNebula
            from langchain_core.messages import HumanMessage

            chat = ChatNebula(model="command", max_new_tokens=256, temperature=0.75)

            messages = [HumanMessage(content="knock knock")]
            chat.invoke(messages)
    """

    model: Optional[str] = None
    """
    model name for above provider
    """

    max_new_tokens: int = 1024
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = 0
    """A non-negative float that tunes the degree of randomness in generation."""

    streaming: bool = False
    """Whether to stream the results."""

    fallback_providers: Optional[str] = None
    """Providers in this will be used as fallback if the call to provider fails."""

    nebula_api_url: str = "https://api-nebula.symbl.ai"

    nebula_api_key: Optional[SecretStr] = Field(None, description="Nebula API Token")

    class Config:
        """Configuration for this pydantic object."""
        allow_population_by_field_name = True
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key exists in environment."""
        values["nebula_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "nebula_api_key", "NEBULA_API_KEY")
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "nebula-chat"
    
    @property
    def _api_key(self) -> str:
        if self.nebula_api_key:
            return self.nebula_api_key.get_secret_value()
        return ""

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Call out to Nebula's chat endpoint."""
        url = f"{self.nebula_api_url}/v1/model/chat/streaming"
        headers = {
            "ApiKey": self._api_key,
            'Content-Type': 'application/json',
        }
        formatted_data = _format_nebula_messages(messages=messages)
        payload: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}
        payload = json.dumps(payload)
        
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()

        for chunk_response in response.iter_lines():
            # print("CHUNK RESPONSEEEEEEEE", chunk_response)
            chunk_decoded = chunk_response.decode()[6:]
            try:
                chunk = json.loads(chunk_decoded)
            except:
                continue
            token = chunk["delta"]
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
        url = f"{self.nebula_api_url}/v1/model/chat/streaming"
        headers = {
            "ApiKey": self._api_key,
            'Content-Type': 'application/json'
        }
        formatted_data = _format_nebula_messages(messages=messages)
        payload: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}
        payload = json.dumps(payload)

        async with ClientSession() as session:
            async with session.post(url, data=payload, headers=headers) as response:
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
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        url = f"{self.nebula_api_url}/v1/model/chat"
        headers = {
            "ApiKey": self._api_key,
            'Content-Type': 'application/json'
        }
        formatted_data = _format_nebula_messages(messages=messages)
        payload: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}
        payload = json.dumps(payload)
            
        response = requests.request("POST", url, headers=headers, data=payload)
        response.raise_for_status()
        data = response.json()

        return ChatResult(
            generations=[
                ChatGeneration(
                    message=AIMessage(content=data["messages"])
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

        url = f"{self.nebula_api_url}/v1/model/chat"
        headers = {
            "ApiKey": self._api_key,
            'Content-Type': 'application/json'
        }
        formatted_data = _format_nebula_messages(messages=messages)
        payload: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "fallback_providers": self.fallback_providers,
            **formatted_data,
            **kwargs,
        }

        payload = {k: v for k, v in payload.items() if v is not None}
        payload = json.dumps(payload)

        async with ClientSession() as session:
            async with session.post(url, data=payload, headers=headers) as response:
                response.raise_for_status()
                response = requests.request("POST", url, headers=headers, data=payload)
                data = await response.json()

                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content=data["messages"]
                            )
                        )
                    ],
                    llm_output=data,
                )
