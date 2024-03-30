import os
import re
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple, Union

import anthropic
from langchain_core._api.deprecation import deprecated
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
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.utils import (
    build_extra_kwargs,
    convert_to_secret_str,
    get_pydantic_field_names,
)

_message_type_lookups = {"human": "user", "ai": "assistant"}


def _format_image(image_url: str) -> Dict:
    """
    Formats an image of format data:image/jpeg;base64,{b64_string}
    to a dict for anthropic api

    {
      "type": "base64",
      "media_type": "image/jpeg",
      "data": "/9j/4AAQSkZJRg...",
    }

    And throws an error if it's not a b64 image
    """
    regex = r"^data:(?P<media_type>image/.+);base64,(?P<data>.+)$"
    match = re.match(regex, image_url)
    if match is None:
        raise ValueError(
            "Anthropic only supports base64-encoded images currently."
            " Example: data:image/png;base64,'/9j/4AAQSk'..."
        )
    return {
        "type": "base64",
        "media_type": match.group("media_type"),
        "data": match.group("data"),
    }


def _format_messages(messages: List[BaseMessage]) -> Tuple[Optional[str], List[Dict]]:
    """Format messages for anthropic."""

    """
    [
                {
                    "role": _message_type_lookups[m.type],
                    "content": [_AnthropicMessageContent(text=m.content).dict()],
                }
                for m in messages
            ]
    """
    system: Optional[str] = None
    formatted_messages: List[Dict] = []
    for i, message in enumerate(messages):
        if message.type == "system":
            if i != 0:
                raise ValueError("System message must be at beginning of message list.")
            if not isinstance(message.content, str):
                raise ValueError(
                    "System message must be a string, "
                    f"instead was: {type(message.content)}"
                )
            system = message.content
            continue

        role = _message_type_lookups[message.type]
        content: Union[str, List[Dict]]

        if not isinstance(message.content, str):
            # parse as dict
            assert isinstance(
                message.content, list
            ), "Anthropic message content must be str or list of dicts"

            # populate content
            content = []
            for item in message.content:
                if isinstance(item, str):
                    content.append(
                        {
                            "type": "text",
                            "text": item,
                        }
                    )
                elif isinstance(item, dict):
                    if "type" not in item:
                        raise ValueError("Dict content item must have a type key")
                    if item["type"] == "image_url":
                        # convert format
                        source = _format_image(item["image_url"]["url"])
                        content.append(
                            {
                                "type": "image",
                                "source": source,
                            }
                        )
                    else:
                        content.append(item)
                else:
                    raise ValueError(
                        f"Content items must be str or dict, instead was: {type(item)}"
                    )
        else:
            content = message.content

        formatted_messages.append(
            {
                "role": role,
                "content": content,
            }
        )
    return system, formatted_messages


class ChatAnthropic(BaseChatModel):
    """Anthropic chat model.

    To use, you should have the environment variable ``ANTHROPIC_API_KEY``
    set with your API key, or pass it as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(model='claude-3-opus-20240229')
    """

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    _client: anthropic.Client = Field(default=None)
    _async_client: anthropic.AsyncClient = Field(default=None)

    model: str = Field(alias="model_name")
    """Model name to use."""

    max_tokens: int = Field(default=1024, alias="max_tokens_to_sample")
    """Denotes the number of tokens to predict per generation."""

    temperature: Optional[float] = None
    """A non-negative float that tunes the degree of randomness in generation."""

    top_k: Optional[int] = None
    """Number of most likely tokens to consider at each step."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    default_request_timeout: Optional[float] = None
    """Timeout for requests to Anthropic Completion API. Default is 600 seconds."""

    anthropic_api_url: str = "https://api.anthropic.com"

    anthropic_api_key: Optional[SecretStr] = None

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)

    streaming: bool = False
    """Whether to use streaming or not."""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "anthropic-chat"

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
        anthropic_api_key = convert_to_secret_str(
            values.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY") or ""
        )
        values["anthropic_api_key"] = anthropic_api_key
        api_key = anthropic_api_key.get_secret_value()
        api_url = (
            values.get("anthropic_api_url")
            or os.environ.get("ANTHROPIC_API_URL")
            or "https://api.anthropic.com"
        )
        values["anthropic_api_url"] = api_url
        values["_client"] = anthropic.Client(api_key=api_key, base_url=api_url)
        values["_async_client"] = anthropic.AsyncClient(
            api_key=api_key, base_url=api_url
        )
        return values

    def _format_params(
        self,
        *,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Dict,
    ) -> Dict:
        # get system prompt if any
        system, formatted_messages = _format_messages(messages)
        rtn = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": formatted_messages,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
            "stop_sequences": stop,
            "system": system,
            **self.model_kwargs,
        }
        rtn = {k: v for k, v in rtn.items() if v is not None}

        return rtn

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        with self._client.messages.stream(**params) as stream:
            for text in stream.text_stream:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        async with self._async_client.messages.stream(**params) as stream:
            async for text in stream.text_stream:
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=text))
                if run_manager:
                    await run_manager.on_llm_new_token(text, chunk=chunk)
                yield chunk

    def _format_output(self, data: Any, **kwargs: Any) -> ChatResult:
        data_dict = data.model_dump()
        content = data_dict["content"]
        llm_output = {
            k: v for k, v in data_dict.items() if k not in ("content", "role", "type")
        }
        return ChatResult(
            generations=[ChatGeneration(message=AIMessage(content=content[0]["text"]))],
            llm_output=llm_output,
        )

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
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        data = self._client.messages.create(**params)
        return self._format_output(data, **kwargs)

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
        params = self._format_params(messages=messages, stop=stop, **kwargs)
        data = await self._async_client.messages.create(**params)
        return self._format_output(data, **kwargs)


@deprecated(since="0.1.0", removal="0.2.0", alternative="ChatAnthropic")
class ChatAnthropicMessages(ChatAnthropic):
    pass
