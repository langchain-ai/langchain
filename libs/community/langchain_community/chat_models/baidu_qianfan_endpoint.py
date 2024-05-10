import logging
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Type,
    Union,
    cast,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, SecretStr, root_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool

logger = logging.getLogger(__name__)


def convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a message to a dictionary that can be passed to the API."""
    message_dict: Dict[str, Any]
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
            # If function call only, content is None not empty string
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, FunctionMessage):
        message_dict = {
            "role": "function",
            "content": message.content,
            "name": message.name,
        }
    else:
        raise TypeError(f"Got unknown type {message}")

    return message_dict


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> AIMessage:
    content = _dict.get("result", "") or ""
    additional_kwargs: Mapping[str, Any] = {}
    if _dict.get("function_call"):
        additional_kwargs = {"function_call": dict(_dict["function_call"])}
        if "thoughts" in additional_kwargs["function_call"]:
            # align to api sample, which affects the llm function_call output
            additional_kwargs["function_call"].pop("thoughts")

    additional_kwargs = {**_dict.get("body", {}), **additional_kwargs}
    return AIMessage(
        content=content,
        additional_kwargs=dict(
            finish_reason=additional_kwargs.get("finish_reason", ""),
            request_id=additional_kwargs["id"],
            object=additional_kwargs.get("object", ""),
            search_info=additional_kwargs.get("search_info", []),
            function_call=additional_kwargs.get("function_call", {}),
            tool_calls=[
                {
                    "type": "function",
                    "function": additional_kwargs.get("function_call", {}),
                }
            ],
        ),
    )


class QianfanChatEndpoint(BaseChatModel):
    """Baidu Qianfan chat models.

    To use, you should have the ``qianfan`` python package installed, and
    the environment variable ``qianfan_ak`` and ``qianfan_sk`` set with your
    API key and Secret Key.

    ak, sk are required parameters
    which you could get from  https://cloud.baidu.com/product/wenxinworkshop

    Example:
        .. code-block:: python

            from langchain_community.chat_models import QianfanChatEndpoint
            qianfan_chat = QianfanChatEndpoint(model="ERNIE-Bot",
                endpoint="your_endpoint", qianfan_ak="your_ak", qianfan_sk="your_sk")
    """

    init_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """init kwargs for qianfan client init, such as `query_per_second` which is 
        associated with qianfan resource object to limit QPS"""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """extra params for model invoke using with `do`."""

    client: Any

    qianfan_ak: Optional[SecretStr] = None
    qianfan_sk: Optional[SecretStr] = None

    streaming: Optional[bool] = False
    """Whether to stream the results or not."""

    request_timeout: Optional[int] = Field(60, alias="timeout")
    """request timeout for chat http requests"""

    top_p: Optional[float] = 0.8
    temperature: Optional[float] = 0.95
    penalty_score: Optional[float] = 1
    """Model params, only supported in ERNIE-Bot and ERNIE-Bot-turbo.
    In the case of other model, passing these params will not affect the result.
    """

    model: str = "ERNIE-Bot-turbo"
    """Model name.
    you could get from https://cloud.baidu.com/doc/WENXINWORKSHOP/s/Nlks5zkzu
    
    preset models are mapping to an endpoint.
    `model` will be ignored if `endpoint` is set.
    Default is ERNIE-Bot-turbo.
    """

    endpoint: Optional[str] = None
    """Endpoint of the Qianfan LLM, required if custom model used."""

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        values["qianfan_ak"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "qianfan_ak",
                "QIANFAN_AK",
                default="",
            )
        )
        values["qianfan_sk"] = convert_to_secret_str(
            get_from_dict_or_env(
                values,
                "qianfan_sk",
                "QIANFAN_SK",
                default="",
            )
        )
        params = {
            **values.get("init_kwargs", {}),
            "model": values["model"],
            "stream": values["streaming"],
        }
        if values["qianfan_ak"].get_secret_value() != "":
            params["ak"] = values["qianfan_ak"].get_secret_value()
        if values["qianfan_sk"].get_secret_value() != "":
            params["sk"] = values["qianfan_sk"].get_secret_value()
        if values["endpoint"] is not None and values["endpoint"] != "":
            params["endpoint"] = values["endpoint"]
        try:
            import qianfan

            values["client"] = qianfan.ChatCompletion(**params)
        except ImportError:
            raise ImportError(
                "qianfan package not found, please install it with "
                "`pip install qianfan`"
            )
        return values

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            **{"endpoint": self.endpoint, "model": self.model},
            **super()._identifying_params,
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat_model."""
        return "baidu-qianfan-chat"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Qianfan API."""
        normal_params = {
            "model": self.model,
            "endpoint": self.endpoint,
            "stream": self.streaming,
            "request_timeout": self.request_timeout,
            "top_p": self.top_p,
            "temperature": self.temperature,
            "penalty_score": self.penalty_score,
        }

        return {**normal_params, **self.model_kwargs}

    def _convert_prompt_msg_params(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Converts a list of messages into a dictionary containing the message content
        and default parameters.

        Args:
            messages (List[BaseMessage]): The list of messages.
            **kwargs (Any): Optional arguments to add additional parameters to the
            resulting dictionary.

        Returns:
            Dict[str, Any]: A dictionary containing the message content and default
            parameters.

        """
        messages_dict: Dict[str, Any] = {
            "messages": [
                convert_message_to_dict(m)
                for m in messages
                if not isinstance(m, SystemMessage)
            ]
        }
        for i in [i for i, m in enumerate(messages) if isinstance(m, SystemMessage)]:
            if "system" not in messages_dict:
                messages_dict["system"] = ""
            messages_dict["system"] += cast(str, messages[i].content) + "\n"

        return {
            **messages_dict,
            **self._default_params,
            **kwargs,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to an qianfan models endpoint for each generation with a prompt.
        Args:
            messages: The messages to pass into the model.
            stop: Optional list of stop words to use when generating.
        Returns:
            The string generated by the model.

        Example:
            .. code-block:: python
                response = qianfan_model.invoke("Tell me a joke.")
        """
        if self.streaming:
            completion = ""
            token_usage = {}
            chat_generation_info: Dict = {}
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                chat_generation_info = (
                    chunk.generation_info
                    if chunk.generation_info is not None
                    else chat_generation_info
                )
                completion += chunk.text
            lc_msg = AIMessage(content=completion, additional_kwargs={})
            gen = ChatGeneration(
                message=lc_msg,
                generation_info=dict(finish_reason="stop"),
            )
            return ChatResult(
                generations=[gen],
                llm_output={
                    "token_usage": chat_generation_info.get("usage", {}),
                    "model_name": self.model,
                },
            )
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params["stop"] = stop
        response_payload = self.client.do(**params)
        lc_msg = _convert_dict_to_message(response_payload)
        gen = ChatGeneration(
            message=lc_msg,
            generation_info={
                "finish_reason": "stop",
                **response_payload.get("body", {}),
            },
        )
        token_usage = response_payload.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return ChatResult(generations=[gen], llm_output=llm_output)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        if self.streaming:
            completion = ""
            token_usage = {}
            chat_generation_info: Dict = {}
            async for chunk in self._astream(messages, stop, run_manager, **kwargs):
                chat_generation_info = (
                    chunk.generation_info
                    if chunk.generation_info is not None
                    else chat_generation_info
                )
                completion += chunk.text

            lc_msg = AIMessage(content=completion, additional_kwargs={})
            gen = ChatGeneration(
                message=lc_msg,
                generation_info=dict(finish_reason="stop"),
            )
            return ChatResult(
                generations=[gen],
                llm_output={
                    "token_usage": chat_generation_info.get("usage", {}),
                    "model_name": self.model,
                },
            )
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params["stop"] = stop
        response_payload = await self.client.ado(**params)
        lc_msg = _convert_dict_to_message(response_payload)
        generations = []
        gen = ChatGeneration(
            message=lc_msg,
            generation_info={
                "finish_reason": "stop",
                **response_payload.get("body", {}),
            },
        )
        generations.append(gen)
        token_usage = response_payload.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params["stop"] = stop
        params["stream"] = True
        for res in self.client.do(**params):
            if res:
                msg = _convert_dict_to_message(res)
                additional_kwargs = msg.additional_kwargs.get("function_call", {})
                chunk = ChatGenerationChunk(
                    text=res["result"],
                    message=AIMessageChunk(
                        content=msg.content,
                        role="assistant",
                        additional_kwargs=additional_kwargs,
                    ),
                    generation_info=msg.additional_kwargs,
                )
                if run_manager:
                    run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        params = self._convert_prompt_msg_params(messages, **kwargs)
        params["stop"] = stop
        params["stream"] = True
        async for res in await self.client.ado(**params):
            if res:
                msg = _convert_dict_to_message(res)
                additional_kwargs = msg.additional_kwargs.get("function_call", {})
                chunk = ChatGenerationChunk(
                    text=res["result"],
                    message=AIMessageChunk(
                        content=msg.content,
                        role="assistant",
                        additional_kwargs=additional_kwargs,
                    ),
                    generation_info=msg.additional_kwargs,
                )
                if run_manager:
                    await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                yield chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.
        """

        formatted_tools = [convert_to_openai_tool(tool)["function"] for tool in tools]
        return super().bind(functions=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]],
        *,
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
                must match the OpenAI function-calling spec.
            include_raw: If False then only the parsed structured output is returned. If
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

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_mistralai import QianfanChatEndpoint
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = QianfanChatEndpoint(endpoint="ernie-3.5-8k-0329")
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Function-calling, Pydantic schema (method="function_calling", include_raw=True):
            .. code-block:: python

                from langchain_mistralai import QianfanChatEndpoint
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = QianfanChatEndpoint(endpoint="ernie-3.5-8k-0329")
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: Function-calling, dict schema (method="function_calling", include_raw=False):
            .. code-block:: python

                from langchain_mistralai import QianfanChatEndpoint
                from langchain_core.pydantic_v1 import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = QianfanChatEndpoint(endpoint="ernie-3.5-8k-0329")
                structured_llm = llm.with_structured_output(dict_schema)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume and density of the two substances differ.'
                # }

        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        llm = self.bind_tools([schema])
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[schema], first_tool_only=True
            )
        else:
            key_name = convert_to_openai_tool(schema)["function"]["name"]
            output_parser = JsonOutputKeyToolsParser(
                key_name=key_name, first_tool_only=True
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
