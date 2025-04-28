import json
import re
import uuid
from abc import ABC, abstractmethod
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
)

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    generate_from_stream,
)
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.messages.tool import ToolCallChunk
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_function
from pydantic import BaseModel, ConfigDict

from langchain_community.llms.oci_generative_ai import OCIGenAIBase
from langchain_community.llms.utils import enforce_stop_tokens

CUSTOM_ENDPOINT_PREFIX = "ocid1.generativeaiendpoint"

JSON_TO_PYTHON_TYPES = {
    "string": "str",
    "number": "float",
    "boolean": "bool",
    "integer": "int",
    "array": "List",
    "object": "Dict",
    "any": "any",
}


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


def _remove_signature_from_tool_description(name: str, description: str) -> str:
    """
    Removes the `{name}{signature} - ` prefix and Args: section from tool description.
    The signature is usually present for tools created with the @tool decorator,
    whereas the Args: section may be present in function doc blocks.
    """
    description = re.sub(rf"^{name}\(.*?\) -(?:> \w+? -)? ", "", description)
    description = re.sub(r"(?s)(?:\n?\n\s*?)?Args:.*$", "", description)
    return description


def _format_oci_tool_calls(
    tool_calls: Optional[List[Any]] = None,
) -> List[Dict]:
    """
    Formats a OCI GenAI API response into the tool call format used in Langchain.
    """
    if not tool_calls:
        return []

    formatted_tool_calls = []
    for tool_call in tool_calls:
        formatted_tool_calls.append(
            {
                "id": uuid.uuid4().hex[:],
                "function": {
                    "name": tool_call.name,
                    "arguments": json.dumps(tool_call.parameters),
                },
                "type": "function",
            }
        )
    return formatted_tool_calls


def _convert_oci_tool_call_to_langchain(tool_call: Any) -> ToolCall:
    """Convert a OCI GenAI tool call into langchain_core.messages.ToolCall"""
    _id = uuid.uuid4().hex[:]
    return ToolCall(name=tool_call.name, args=tool_call.parameters, id=_id)


class Provider(ABC):
    @property
    @abstractmethod
    def stop_sequence_key(self) -> str: ...

    @abstractmethod
    def chat_response_to_text(self, response: Any) -> str: ...

    @abstractmethod
    def chat_stream_to_text(self, event_data: Dict) -> str: ...

    @abstractmethod
    def is_chat_stream_end(self, event_data: Dict) -> bool: ...

    @abstractmethod
    def chat_generation_info(self, response: Any) -> Dict[str, Any]: ...

    @abstractmethod
    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]: ...

    @abstractmethod
    def get_role(self, message: BaseMessage) -> str: ...

    @abstractmethod
    def messages_to_oci_params(
        self, messages: Any, **kwargs: Any
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def convert_to_oci_tool(
        self,
        tool: Union[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
    ) -> Dict[str, Any]: ...


class CohereProvider(Provider):
    stop_sequence_key: str = "stop_sequences"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.oci_chat_request = models.CohereChatRequest
        self.oci_tool = models.CohereTool
        self.oci_tool_param = models.CohereParameterDefinition
        self.oci_tool_result = models.CohereToolResult
        self.oci_tool_call = models.CohereToolCall
        self.oci_chat_message = {
            "USER": models.CohereUserMessage,
            "CHATBOT": models.CohereChatBotMessage,
            "SYSTEM": models.CohereSystemMessage,
            "TOOL": models.CohereToolMessage,
        }
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_COHERE

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        if "text" in event_data:
            if "finishReason" in event_data or "toolCalls" in event_data:
                return ""
            else:
                return event_data["text"]
        else:
            return ""

    def is_chat_stream_end(self, event_data: Dict) -> bool:
        return "finishReason" in event_data

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        generation_info: Dict[str, Any] = {
            "documents": response.data.chat_response.documents,
            "citations": response.data.chat_response.citations,
            "search_queries": response.data.chat_response.search_queries,
            "is_search_required": response.data.chat_response.is_search_required,
            "finish_reason": response.data.chat_response.finish_reason,
        }
        if response.data.chat_response.tool_calls:
            # Only populate tool_calls when 1) present on the response and
            #  2) has one or more calls.
            generation_info["tool_calls"] = _format_oci_tool_calls(
                response.data.chat_response.tool_calls
            )

        return generation_info

    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        generation_info: Dict[str, Any] = {
            "documents": event_data.get("documents"),
            "citations": event_data.get("citations"),
            "finish_reason": event_data.get("finishReason"),
        }
        if "toolCalls" in event_data:
            generation_info["tool_calls"] = []
            for tool_call in event_data["toolCalls"]:
                generation_info["tool_calls"].append(
                    {
                        "id": uuid.uuid4().hex[:],
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call["parameters"]),
                        },
                        "type": "function",
                    }
                )

        generation_info = {k: v for k, v in generation_info.items() if v is not None}

        return generation_info

    def get_role(self, message: BaseMessage) -> str:
        if isinstance(message, HumanMessage):
            return "USER"
        elif isinstance(message, AIMessage):
            return "CHATBOT"
        elif isinstance(message, SystemMessage):
            return "SYSTEM"
        elif isinstance(message, ToolMessage):
            return "TOOL"
        else:
            raise ValueError(f"Got unknown type {message}")

    def messages_to_oci_params(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        is_force_single_step = kwargs.get("is_force_single_step") or False

        oci_chat_history = []

        for msg in messages[:-1]:
            if self.get_role(msg) == "USER" or self.get_role(msg) == "SYSTEM":
                oci_chat_history.append(
                    self.oci_chat_message[self.get_role(msg)](message=msg.content)
                )
            elif isinstance(msg, AIMessage):
                if msg.tool_calls and is_force_single_step:
                    continue
                tool_calls = (
                    [
                        self.oci_tool_call(name=tc["name"], parameters=tc["args"])
                        for tc in msg.tool_calls
                    ]
                    if msg.tool_calls
                    else None
                )
                msg_content = msg.content if msg.content else " "
                oci_chat_history.append(
                    self.oci_chat_message[self.get_role(msg)](
                        message=msg_content, tool_calls=tool_calls
                    )
                )

        # Get the messages for the current chat turn
        current_chat_turn_messages = []
        for message in messages[::-1]:
            current_chat_turn_messages.append(message)
            if isinstance(message, HumanMessage):
                break
        current_chat_turn_messages = current_chat_turn_messages[::-1]

        oci_tool_results: Union[List[Any], None] = []
        for message in current_chat_turn_messages:
            if isinstance(message, ToolMessage):
                tool_message = message
                previous_ai_msgs = [
                    message
                    for message in current_chat_turn_messages
                    if isinstance(message, AIMessage) and message.tool_calls
                ]
                if previous_ai_msgs:
                    previous_ai_msg = previous_ai_msgs[-1]
                    for lc_tool_call in previous_ai_msg.tool_calls:
                        if lc_tool_call["id"] == tool_message.tool_call_id:
                            tool_result = self.oci_tool_result()
                            tool_result.call = self.oci_tool_call(
                                name=lc_tool_call["name"],
                                parameters=lc_tool_call["args"],
                            )
                            tool_result.outputs = [{"output": tool_message.content}]
                            oci_tool_results.append(tool_result)

        if not oci_tool_results:
            oci_tool_results = None

        message_str = "" if oci_tool_results else messages[-1].content

        oci_params = {
            "message": message_str,
            "chat_history": oci_chat_history,
            "tool_results": oci_tool_results,
            "api_format": self.chat_api_format,
        }

        return {k: v for k, v in oci_params.items() if v is not None}

    def convert_to_oci_tool(
        self,
        tool: Union[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
    ) -> Dict[str, Any]:
        """
        Convert a BaseTool instance, JSON schema dict, or BaseModel type to a OCI tool.
        """
        if isinstance(tool, BaseTool):
            return self.oci_tool(
                name=tool.name,
                description=_remove_signature_from_tool_description(
                    tool.name, tool.description
                ),
                parameter_definitions={
                    p_name: self.oci_tool_param(
                        description=p_def.get("description")
                        if "description" in p_def
                        else "",
                        type=JSON_TO_PYTHON_TYPES.get(
                            p_def.get("type"), p_def.get("type", "any")
                        ),
                        is_required="default" not in p_def,
                    )
                    for p_name, p_def in tool.args.items()
                },
            )
        elif isinstance(tool, dict):
            if not all(k in tool for k in ("title", "description", "properties")):
                raise ValueError(
                    "Unsupported dict type. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
                )
            return self.oci_tool(
                name=tool.get("title"),
                description=tool.get("description"),
                parameter_definitions={
                    p_name: self.oci_tool_param(
                        description=p_def.get("description"),
                        type=JSON_TO_PYTHON_TYPES.get(
                            p_def.get("type"), p_def.get("type", "any")
                        ),
                        is_required="default" not in p_def,
                    )
                    for p_name, p_def in tool.get("properties", {}).items()
                },
            )
        elif (isinstance(tool, type) and issubclass(tool, BaseModel)) or callable(tool):
            as_json_schema_function = convert_to_openai_function(tool)
            parameters = as_json_schema_function.get("parameters", {})
            properties = parameters.get("properties", {})
            return self.oci_tool(
                name=as_json_schema_function.get("name"),
                description=as_json_schema_function.get(
                    "description",
                    as_json_schema_function.get("name"),
                ),
                parameter_definitions={
                    p_name: self.oci_tool_param(
                        description=p_def.get("description"),
                        type=JSON_TO_PYTHON_TYPES.get(
                            p_def.get("type"), p_def.get("type", "any")
                        ),
                        is_required=p_name in parameters.get("required", []),
                    )
                    for p_name, p_def in properties.items()
                },
            )
        else:
            raise ValueError(
                f"Unsupported tool type {type(tool)}. Tool must be passed in as a BaseTool instance, JSON schema dict, or BaseModel type."  # noqa: E501
            )


class MetaProvider(Provider):
    stop_sequence_key: str = "stop"

    def __init__(self) -> None:
        from oci.generative_ai_inference import models

        self.oci_chat_request = models.GenericChatRequest
        self.oci_chat_message = {
            "USER": models.UserMessage,
            "SYSTEM": models.SystemMessage,
            "ASSISTANT": models.AssistantMessage,
        }
        self.oci_chat_message_content = models.ChatContent
        self.oci_chat_message_text_content = models.TextContent
        self.oci_chat_message_image_content = models.ImageContent
        self.oci_chat_message_image_url = models.ImageUrl
        self.chat_api_format = models.BaseChatRequest.API_FORMAT_GENERIC

    def chat_response_to_text(self, response: Any) -> str:
        return response.data.chat_response.choices[0].message.content[0].text

    def chat_stream_to_text(self, event_data: Dict) -> str:
        return event_data["message"]["content"][0]["text"]

    def is_chat_stream_end(self, event_data: Dict) -> bool:
        return "message" not in event_data

    def chat_generation_info(self, response: Any) -> Dict[str, Any]:
        return {
            "finish_reason": response.data.chat_response.choices[0].finish_reason,
            "time_created": str(response.data.chat_response.time_created),
        }

    def chat_stream_generation_info(self, event_data: Dict) -> Dict[str, Any]:
        return {
            "finish_reason": event_data["finishReason"],
        }

    def get_role(self, message: BaseMessage) -> str:
        # meta only supports alternating user/assistant roles
        if isinstance(message, HumanMessage):
            return "USER"
        elif isinstance(message, AIMessage):
            return "ASSISTANT"
        elif isinstance(message, SystemMessage):
            return "SYSTEM"
        else:
            raise ValueError(f"Got unknown type {message}")

    def messages_to_oci_params(
        self, messages: List[BaseMessage], **kwargs: Any
    ) -> Dict[str, Any]:
        """Convert LangChain messages to OCI chat parameters.

        Args:
            messages: List of LangChain BaseMessage objects
            **kwargs: Additional keyword arguments

        Returns:
            Dict containing OCI chat parameters

        Raises:
            ValueError: If message content is invalid
        """
        oci_messages = []

        for message in messages:
            content = self._process_message_content(message.content)
            oci_message = self.oci_chat_message[self.get_role(message)](content=content)
            oci_messages.append(oci_message)

        return {
            "messages": oci_messages,
            "api_format": self.chat_api_format,
            "top_k": -1,
        }

    def _process_message_content(
        self, content: Union[str, List[Union[str, Dict]]]
    ) -> List[Any]:
        """Process message content into OCI chat content format.

        Args:
            content: Message content as string or list

        Returns:
            List of OCI chat content objects

        Raises:
            ValueError: If content format is invalid
        """
        if isinstance(content, str):
            return [self.oci_chat_message_text_content(text=content)]

        if not isinstance(content, list):
            raise ValueError("Message content must be str or list of items")

        processed_content = []
        for item in content:
            if isinstance(item, str):
                processed_content.append(self.oci_chat_message_text_content(text=item))
                continue

            if not isinstance(item, dict):
                raise ValueError(
                    f"Content items must be str or dict, got: {type(item)}"
                )

            if "type" not in item:
                raise ValueError("Dict content item must have a type key")

            if item["type"] == "image_url":
                processed_content.append(
                    self.oci_chat_message_image_content(
                        image_url=self.oci_chat_message_image_url(
                            url=item["image_url"]["url"]
                        )
                    )
                )
            elif item["type"] == "text":
                processed_content.append(
                    self.oci_chat_message_text_content(text=item["text"])
                )
            else:
                raise ValueError(f"Unsupported content type: {item['type']}")

        return processed_content

    def convert_to_oci_tool(
        self,
        tool: Union[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
    ) -> Dict[str, Any]:
        raise NotImplementedError("Tools not supported for Meta models")


class ChatOCIGenAI(BaseChatModel, OCIGenAIBase):
    """ChatOCIGenAI chat model integration.

    Setup:
      Install ``langchain-community`` and the ``oci`` sdk.

      .. code-block:: bash

          pip install -U langchain-community oci

    Key init args — completion params:
        model_id: str
            Id of the OCIGenAI chat model to use, e.g., cohere.command-r-16k.
        is_stream: bool
            Whether to stream back partial progress
        model_kwargs: Optional[Dict]
            Keyword arguments to pass to the specific model used, e.g., temperature, max_tokens.

    Key init args — client params:
        service_endpoint: str
            The endpoint URL for the OCIGenAI service, e.g., https://inference.generativeai.us-chicago-1.oci.oraclecloud.com.
        compartment_id: str
            The compartment OCID.
        auth_type: str
            The authentication type to use, e.g., API_KEY (default), SECURITY_TOKEN, INSTANCE_PRINCIPAL, RESOURCE_PRINCIPAL.
        auth_profile: Optional[str]
            The name of the profile in ~/.oci/config, if not specified , DEFAULT will be used.
        auth_file_location: Optional[str]
            Path to the config file, If not specified, ~/.oci/config will be used.
        provider: str
            Provider name of the model. Default to None, will try to be derived from the model_id otherwise, requires user input.
    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatOCIGenAI

            chat = ChatOCIGenAI(
                model_id="cohere.command-r-16k",
                service_endpoint="https://inference.generativeai.us-chicago-1.oci.oraclecloud.com",
                compartment_id="MY_OCID",
                model_kwargs={"temperature": 0.7, "max_tokens": 500},
            )

    Invoke:
        .. code-block:: python
            messages = [
                SystemMessage(content="your are an AI assistant."),
                AIMessage(content="Hi there human!"),
                HumanMessage(content="tell me a joke."),
            ]
            response = chat.invoke(messages)

    Stream:
        .. code-block:: python

        for r in chat.stream(messages):
            print(r.content, end="", flush=True)

    Response metadata
        .. code-block:: python

        response = chat.invoke(messages)
        print(response.response_metadata)

    """  # noqa: E501

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_generative_ai_chat"

    @property
    def _provider_map(self) -> Mapping[str, Any]:
        """Get the provider map"""
        return {
            "cohere": CohereProvider(),
            "meta": MetaProvider(),
        }

    @property
    def _provider(self) -> Any:
        """Get the internal provider object"""
        return self._get_provider(provider_map=self._provider_map)

    def _prepare_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]],
        stream: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        try:
            from oci.generative_ai_inference import models

        except ImportError as ex:
            raise ModuleNotFoundError(
                "Could not import oci python package. "
                "Please make sure you have the oci package installed."
            ) from ex

        oci_params = self._provider.messages_to_oci_params(messages, **kwargs)

        oci_params["is_stream"] = stream
        _model_kwargs = self.model_kwargs or {}

        if stop is not None:
            _model_kwargs[self._provider.stop_sequence_key] = stop

        chat_params = {**_model_kwargs, **kwargs, **oci_params}

        if not self.model_id:
            raise ValueError("Model ID is required to chat")

        if self.model_id.startswith(CUSTOM_ENDPOINT_PREFIX):
            serving_mode = models.DedicatedServingMode(endpoint_id=self.model_id)
        else:
            serving_mode = models.OnDemandServingMode(model_id=self.model_id)

        request = models.ChatDetails(
            compartment_id=self.compartment_id,
            serving_mode=serving_mode,
            chat_request=self._provider.oci_chat_request(**chat_params),
        )

        return request

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        formatted_tools = [self._provider.convert_to_oci_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OCI Generative AI function-calling spec.
            method:
                The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OCI function and the returned model will make use of the
                function-calling API. If "json_mode" then Cohere's JSON mode will be
                used. Note that if using "json_mode" then you must include instructions
                for formatting the output into the desired schema into the model call.
            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If include_raw is True then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If include_raw is False then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
            llm = self.bind_tools([schema], **kwargs)
            tool_name = getattr(self._provider.convert_to_oci_tool(schema), "name")
            if is_pydantic_schema:
                output_parser: OutputParserLike = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. "
                f"Expected `function_calling` or `json_mode`."
                f"Received: `{method}`."
            )
        if include_raw:
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            return llm | output_parser

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to a OCIGenAI chat model.

        Args:
            messages: list of LangChain messages
            stop: Optional list of stop words to use.

        Returns:
            LangChain ChatResult

        Example:
            .. code-block:: python

               messages = [
                            HumanMessage(content="hello!"),
                            AIMessage(content="Hi there human!"),
                            HumanMessage(content="Meow!")
                          ]

               response = llm.invoke(messages)
        """
        if self.is_stream:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        request = self._prepare_request(messages, stop=stop, stream=False, **kwargs)
        response = self.client.chat(request)

        content = self._provider.chat_response_to_text(response)

        if stop is not None:
            content = enforce_stop_tokens(content, stop)

        generation_info = self._provider.chat_generation_info(response)

        llm_output = {
            "model_id": response.data.model_id,
            "model_version": response.data.model_version,
            "request_id": response.request_id,
            "content-length": response.headers["content-length"],
        }

        if "tool_calls" in generation_info:
            tool_calls = [
                _convert_oci_tool_call_to_langchain(tool_call)
                for tool_call in response.data.chat_response.tool_calls
            ]
        else:
            tool_calls = []

        message = AIMessage(
            content=content,
            additional_kwargs=generation_info,
            tool_calls=tool_calls,
        )
        return ChatResult(
            generations=[
                ChatGeneration(message=message, generation_info=generation_info)
            ],
            llm_output=llm_output,
        )

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        request = self._prepare_request(messages, stop=stop, stream=True, **kwargs)
        response = self.client.chat(request)

        for event in response.data.events():
            event_data = json.loads(event.data)
            if not self._provider.is_chat_stream_end(event_data):  # still streaming
                delta = self._provider.chat_stream_to_text(event_data)
                chunk = ChatGenerationChunk(message=AIMessageChunk(content=delta))
                if run_manager:
                    run_manager.on_llm_new_token(delta, chunk=chunk)
                yield chunk
            else:  # stream end
                generation_info = self._provider.chat_stream_generation_info(event_data)
                tool_call_chunks = []
                if tool_calls := generation_info.get("tool_calls"):
                    content = self._provider.chat_stream_to_text(event_data)
                    try:
                        tool_call_chunks = [
                            ToolCallChunk(
                                name=tool_call["function"].get("name"),
                                args=tool_call["function"].get("arguments"),
                                id=tool_call.get("id"),
                                index=tool_call.get("index"),
                            )
                            for tool_call in tool_calls
                        ]
                    except KeyError:
                        pass
                else:
                    content = ""
                message = AIMessageChunk(
                    content=content,
                    additional_kwargs=generation_info,
                    tool_call_chunks=tool_call_chunks,
                )
                yield ChatGenerationChunk(
                    message=message,
                    generation_info=generation_info,
                )
