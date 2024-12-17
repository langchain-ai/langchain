import json
import warnings
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessageChunk,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from langchain_community.llms.azureml_endpoint import (
    AzureMLBaseEndpoint,
    AzureMLEndpointApiType,
    ContentFormatterBase,
)


class LlamaContentFormatter(ContentFormatterBase):
    """Content formatter for `LLaMA`."""

    def __init__(self) -> None:
        raise TypeError(
            "`LlamaContentFormatter` is deprecated for chat models. Use "
            "`CustomOpenAIContentFormatter` instead."
        )


class CustomOpenAIChatContentFormatter(ContentFormatterBase):
    """Chat Content formatter for models with OpenAI like API scheme."""

    SUPPORTED_ROLES: List[str] = ["user", "assistant", "system"]

    @staticmethod
    def _convert_message_to_dict(message: BaseMessage) -> Dict:
        """Converts a message to a dict according to a role"""
        content = cast(str, message.content)
        if isinstance(message, HumanMessage):
            return {
                "role": "user",
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        elif isinstance(message, AIMessage):
            return {
                "role": "assistant",
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        elif isinstance(message, SystemMessage):
            return {
                "role": "system",
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        elif (
            isinstance(message, ChatMessage)
            and message.role in CustomOpenAIChatContentFormatter.SUPPORTED_ROLES
        ):
            return {
                "role": message.role,
                "content": ContentFormatterBase.escape_special_characters(content),
            }
        else:
            supported = ",".join(
                [role for role in CustomOpenAIChatContentFormatter.SUPPORTED_ROLES]
            )
            raise ValueError(
                f"""Received unsupported role. 
                Supported roles for the LLaMa Foundation Model: {supported}"""
            )

    @property
    def supported_api_types(self) -> List[AzureMLEndpointApiType]:
        return [AzureMLEndpointApiType.dedicated, AzureMLEndpointApiType.serverless]

    def format_messages_request_payload(
        self,
        messages: List[BaseMessage],
        model_kwargs: Dict,
        api_type: AzureMLEndpointApiType,
    ) -> bytes:
        """Formats the request according to the chosen api"""
        chat_messages = [
            CustomOpenAIChatContentFormatter._convert_message_to_dict(message)
            for message in messages
        ]
        if api_type in [
            AzureMLEndpointApiType.dedicated,
            AzureMLEndpointApiType.realtime,
        ]:
            request_payload = json.dumps(
                {
                    "input_data": {
                        "input_string": chat_messages,
                        "parameters": model_kwargs,
                    }
                }
            )
        elif api_type == AzureMLEndpointApiType.serverless:
            request_payload = json.dumps({"messages": chat_messages, **model_kwargs})
        else:
            raise ValueError(
                f"`api_type` {api_type} is not supported by this formatter"
            )
        return str.encode(request_payload)

    def format_response_payload(
        self,
        output: bytes,
        api_type: AzureMLEndpointApiType = AzureMLEndpointApiType.dedicated,
    ) -> ChatGeneration:
        """Formats response"""
        if api_type in [
            AzureMLEndpointApiType.dedicated,
            AzureMLEndpointApiType.realtime,
        ]:
            try:
                choice = json.loads(output)["output"]
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return ChatGeneration(
                message=AIMessage(
                    content=choice.strip(),
                ),
                generation_info=None,
            )
        if api_type == AzureMLEndpointApiType.serverless:
            try:
                choice = json.loads(output)["choices"][0]
                if not isinstance(choice, dict):
                    raise TypeError(
                        "Endpoint response is not well formed for a chat "
                        "model. Expected `dict` but `{type(choice)}` was received."
                    )
            except (KeyError, IndexError, TypeError) as e:
                raise ValueError(self.format_error_msg.format(api_type=api_type)) from e
            return ChatGeneration(
                message=AIMessage(content=choice["message"]["content"].strip())
                if choice["message"]["role"] == "assistant"
                else BaseMessage(
                    content=choice["message"]["content"].strip(),
                    type=choice["message"]["role"],
                ),
                generation_info=dict(
                    finish_reason=choice.get("finish_reason"),
                    logprobs=choice.get("logprobs"),
                ),
            )
        raise ValueError(f"`api_type` {api_type} is not supported by this formatter")


class LlamaChatContentFormatter(CustomOpenAIChatContentFormatter):
    """Deprecated: Kept for backwards compatibility

    Chat Content formatter for Llama."""

    def __init__(self) -> None:
        super().__init__()
        warnings.warn(
            """`LlamaChatContentFormatter` will be deprecated in the future. 
                Please use `CustomOpenAIChatContentFormatter` instead.  
            """
        )


class MistralChatContentFormatter(LlamaChatContentFormatter):
    """Content formatter for `Mistral`."""

    def format_messages_request_payload(
        self,
        messages: List[BaseMessage],
        model_kwargs: Dict,
        api_type: AzureMLEndpointApiType,
    ) -> bytes:
        """Formats the request according to the chosen api"""
        chat_messages = [self._convert_message_to_dict(message) for message in messages]

        if chat_messages and chat_messages[0]["role"] == "system":
            # Mistral OSS models do not explicitly support system prompts, so we have to
            # stash in the first user prompt
            chat_messages[1]["content"] = (
                chat_messages[0]["content"] + "\n\n" + chat_messages[1]["content"]
            )
            del chat_messages[0]

        if api_type == AzureMLEndpointApiType.realtime:
            request_payload = json.dumps(
                {
                    "input_data": {
                        "input_string": chat_messages,
                        "parameters": model_kwargs,
                    }
                }
            )
        elif api_type == AzureMLEndpointApiType.serverless:
            request_payload = json.dumps({"messages": chat_messages, **model_kwargs})
        else:
            raise ValueError(
                f"`api_type` {api_type} is not supported by this formatter"
            )
        return str.encode(request_payload)


class AzureMLChatOnlineEndpoint(BaseChatModel, AzureMLBaseEndpoint):
    """Azure ML Online Endpoint chat models.

    Example:
        .. code-block:: python
            azure_llm = AzureMLOnlineEndpoint(
                endpoint_url="https://<your-endpoint>.<your_region>.inference.ml.azure.com/v1/chat/completions",
                endpoint_api_type=AzureMLApiType.serverless,
                endpoint_api_key="my-api-key",
                content_formatter=chat_content_formatter,
            )
    """

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"model_kwargs": _model_kwargs},
        }

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "azureml_chat_endpoint"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to an AzureML Managed Online endpoint.
        Args:
            messages: The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.
        Example:
            .. code-block:: python
                response = azureml_model.invoke("Tell me a joke.")
        """
        _model_kwargs = self.model_kwargs or {}
        _model_kwargs.update(kwargs)
        if stop:
            _model_kwargs["stop"] = stop

        request_payload = self.content_formatter.format_messages_request_payload(
            messages, _model_kwargs, self.endpoint_api_type
        )
        response_payload = self.http_client.call(
            body=request_payload, run_manager=run_manager
        )
        generations = self.content_formatter.format_response_payload(
            response_payload, self.endpoint_api_type
        )
        return ChatResult(generations=[generations])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        self.endpoint_url = self.endpoint_url.replace("/chat/completions", "")
        timeout = None if "timeout" not in kwargs else kwargs["timeout"]

        import openai

        params = {}
        client_params = {
            "api_key": self.endpoint_api_key.get_secret_value(),
            "base_url": self.endpoint_url,
            "timeout": timeout,
            "default_headers": None,
            "default_query": None,
            "http_client": None,
        }

        client = openai.OpenAI(**client_params)  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
        message_dicts = [
            CustomOpenAIChatContentFormatter._convert_message_to_dict(m)
            for m in messages
        ]
        params = {"stream": True, "stop": stop, "model": None, **kwargs}

        default_chunk_class = AIMessageChunk
        for chunk in client.chat.completions.create(messages=message_dicts, **params):  # type: ignore[arg-type]
            if not isinstance(chunk, dict):
                chunk = chunk.dict()  # type: ignore[attr-defined]
            if len(chunk["choices"]) == 0:  # type: ignore[call-overload]
                continue
            choice = chunk["choices"][0]  # type: ignore[call-overload]
            chunk = _convert_delta_to_message_chunk(  # type: ignore[assignment]
                choice["delta"],  # type: ignore[arg-type, index]
                default_chunk_class,  # type: ignore[arg-type, index]
            )
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):  # type: ignore[union-attr]
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")  # type: ignore[union-attr]
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = chunk.__class__  # type: ignore[assignment]
            chunk = ChatGenerationChunk(  # type: ignore[assignment]
                message=chunk,  # type: ignore[arg-type]
                generation_info=generation_info or None,  # type: ignore[arg-type]
            )
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)  # type: ignore[attr-defined, arg-type]
            yield chunk  # type: ignore[misc]

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        self.endpoint_url = self.endpoint_url.replace("/chat/completions", "")
        timeout = None if "timeout" not in kwargs else kwargs["timeout"]

        import openai

        params = {}
        client_params = {
            "api_key": self.endpoint_api_key.get_secret_value(),
            "base_url": self.endpoint_url,
            "timeout": timeout,
            "default_headers": None,
            "default_query": None,
            "http_client": None,
        }

        async_client = openai.AsyncOpenAI(**client_params)  # type: ignore[arg-type, arg-type, arg-type, arg-type, arg-type, arg-type]
        message_dicts = [
            CustomOpenAIChatContentFormatter._convert_message_to_dict(m)
            for m in messages
        ]
        params = {"stream": True, "stop": stop, "model": None, **kwargs}

        default_chunk_class = AIMessageChunk
        async for chunk in await async_client.chat.completions.create(  # type: ignore[attr-defined]
            messages=message_dicts,  # type: ignore[arg-type]
            **params,  # type: ignore[arg-type]
        ):
            if not isinstance(chunk, dict):
                chunk = chunk.dict()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            generation_info = {}
            if finish_reason := choice.get("finish_reason"):
                generation_info["finish_reason"] = finish_reason
            logprobs = choice.get("logprobs")
            if logprobs:
                generation_info["logprobs"] = logprobs
            default_chunk_class = chunk.__class__
            chunk = ChatGenerationChunk(
                message=chunk, generation_info=generation_info or None
            )
            if run_manager:
                await run_manager.on_llm_new_token(
                    token=chunk.text, chunk=chunk, logprobs=logprobs
                )
            yield chunk


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    if _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = _dict["tool_calls"]

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(content=content, tool_call_id=_dict["tool_call_id"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)  # type: ignore[call-arg]
