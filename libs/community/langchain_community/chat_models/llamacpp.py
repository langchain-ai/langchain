import json
from operator import itemgetter
from pathlib import Path
from typing import (
    Any,
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
    BaseMessageChunk,
    ChatMessage,
    ChatMessageChunk,
    FunctionMessage,
    FunctionMessageChunk,
    HumanMessage,
    HumanMessageChunk,
    SystemMessage,
    SystemMessageChunk,
    ToolMessage,
    ToolMessageChunk,
)
from langchain_core.messages.tool import InvalidToolCall, ToolCall, ToolCallChunk
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool


class ChatLlamaCpp(BaseChatModel):
    """llama.cpp model.

    To use, you should have the llama-cpp-python library installed, and provide the
    path to the Llama model as a named parameter to the constructor.
    Check out: https://github.com/abetlen/llama-cpp-python

    """

    client: Any  #: :meta private:

    model_path: str
    """The path to the Llama model file."""

    lora_base: Optional[str] = None
    """The path to the Llama LoRA base model."""

    lora_path: Optional[str] = None
    """The path to the Llama LoRA. If None, no LoRa is loaded."""

    n_ctx: int = 512
    """Token context window."""

    n_parts: int = -1
    """Number of parts to split the model into.
    If -1, the number of parts is automatically determined."""

    seed: int = -1
    """Seed. If -1, a random seed is used."""

    f16_kv: bool = True
    """Use half-precision for key/value cache."""

    logits_all: bool = False
    """Return logits for all tokens, not just the last token."""

    vocab_only: bool = False
    """Only load the vocabulary, no weights."""

    use_mlock: bool = False
    """Force system to keep model in RAM."""

    n_threads: Optional[int] = None
    """Number of threads to use.
    If None, the number of threads is automatically determined."""

    n_batch: int = 8
    """Number of tokens to process in parallel.
    Should be a number between 1 and n_ctx."""

    n_gpu_layers: Optional[int] = None
    """Number of layers to be loaded into gpu memory. Default None."""

    suffix: Optional[str] = None
    """A suffix to append to the generated text. If None, no suffix is appended."""

    max_tokens: int = 256
    """The maximum number of tokens to generate."""

    temperature: float = 0.8
    """The temperature to use for sampling."""

    top_p: float = 0.95
    """The top-p value to use for sampling."""

    logprobs: Optional[int] = None
    """The number of logprobs to return. If None, no logprobs are returned."""

    echo: bool = False
    """Whether to echo the prompt."""

    stop: Optional[List[str]] = None
    """A list of strings to stop generation when encountered."""

    repeat_penalty: float = 1.1
    """The penalty to apply to repeated tokens."""

    top_k: int = 40
    """The top-k value to use for sampling."""

    last_n_tokens_size: int = 64
    """The number of tokens to look back when applying the repeat_penalty."""

    use_mmap: bool = True
    """Whether to keep the model loaded in RAM"""

    rope_freq_scale: float = 1.0
    """Scale factor for rope sampling."""

    rope_freq_base: float = 10000.0
    """Base frequency for rope sampling."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Any additional parameters to pass to llama_cpp.Llama."""

    streaming: bool = True
    """Whether to stream the results, token by token."""

    grammar_path: Optional[Union[str, Path]] = None
    """
    grammar_path: Path to the .gbnf file that defines formal grammars
    for constraining model outputs. For instance, the grammar can be used
    to force the model to generate valid JSON or to speak exclusively in emojis. At most
    one of grammar_path and grammar should be passed in.
    """
    grammar: Any = None
    """
    grammar: formal grammar for constraining model outputs. For instance, the grammar 
    can be used to force the model to generate valid JSON or to speak exclusively in 
    emojis. At most one of grammar_path and grammar should be passed in.
    """

    verbose: bool = True
    """Print verbose output to stderr."""

    @root_validator(pre=False, skip_on_failure=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that llama-cpp-python library is installed."""
        try:
            from llama_cpp import Llama, LlamaGrammar
        except ImportError:
            raise ImportError(
                "Could not import llama-cpp-python library. "
                "Please install the llama-cpp-python library to "
                "use this embedding model: pip install llama-cpp-python"
            )

        model_path = values["model_path"]
        model_param_names = [
            "rope_freq_scale",
            "rope_freq_base",
            "lora_path",
            "lora_base",
            "n_ctx",
            "n_parts",
            "seed",
            "f16_kv",
            "logits_all",
            "vocab_only",
            "use_mlock",
            "n_threads",
            "n_batch",
            "use_mmap",
            "last_n_tokens_size",
            "verbose",
        ]
        model_params = {k: values[k] for k in model_param_names}
        # For backwards compatibility, only include if non-null.
        if values["n_gpu_layers"] is not None:
            model_params["n_gpu_layers"] = values["n_gpu_layers"]

        model_params.update(values["model_kwargs"])

        try:
            values["client"] = Llama(model_path, **model_params)
        except Exception as e:
            raise ValueError(
                f"Could not load Llama model from path: {model_path}. "
                f"Received error {e}"
            )

        if values["grammar"] and values["grammar_path"]:
            grammar = values["grammar"]
            grammar_path = values["grammar_path"]
            raise ValueError(
                "Can only pass in one of grammar and grammar_path. Received "
                f"{grammar=} and {grammar_path=}."
            )
        elif isinstance(values["grammar"], str):
            values["grammar"] = LlamaGrammar.from_string(values["grammar"])
        elif values["grammar_path"]:
            values["grammar"] = LlamaGrammar.from_file(values["grammar_path"])
        else:
            pass
        return values

    def _get_parameters(self, stop: Optional[List[str]]) -> Dict[str, Any]:
        """
        Performs sanity check, preparing parameters in format needed by llama_cpp.

        Returns:
            Dictionary containing the combined parameters.
        """

        params = self._default_params

        # llama_cpp expects the "stop" key not this, so we remove it:
        stop_sequences = params.pop("stop_sequences")

        # then sets it as configured, or default to an empty list:
        params["stop"] = stop or stop_sequences or self.stop or []

        return params

    def _create_message_dicts(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        message_dicts = [_convert_message_to_dict(m) for m in messages]

        return message_dicts

    def _create_chat_result(self, response: dict) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = _convert_dict_to_message(res["message"])
            generation_info = dict(finish_reason=res.get("finish_reason"))
            if "logprobs" in res:
                generation_info["logprobs"] = res["logprobs"]
            gen = ChatGeneration(message=message, generation_info=generation_info)
            generations.append(gen)
        token_usage = response.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            # "system_fingerprint": response.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        params = {**self._get_parameters(stop), **kwargs}

        # Check tool_choice is whether available, if yes then run no stream with tool
        # calling
        if self.streaming and not params.get("tool_choice"):
            stream_iter = self._stream(messages, run_manager=run_manager, **kwargs)
            return generate_from_stream(stream_iter)

        message_dicts = self._create_message_dicts(messages)

        response = self.client.create_chat_completion(messages=message_dicts, **params)

        return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        params = {**self._get_parameters(stop), **kwargs}
        message_dicts = self._create_message_dicts(messages)

        result = self.client.create_chat_completion(
            messages=message_dicts, stream=True, **params
        )

        default_chunk_class = AIMessageChunk
        count = 0
        for chunk in result:
            count += 1
            if not isinstance(chunk, dict):
                chunk = chunk.model_dump()
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            if choice["delta"] is None:
                continue
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
                run_manager.on_llm_new_token(chunk.text, chunk=chunk, logprobs=logprobs)
            yield chunk

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[Dict[str, Dict], bool, str]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model

        tool_choice: does not currently support "any", "auto" choices like OpenAI
            tool-calling API. should be a dict of the form to force this tool
            {"type": "function", "function": {"name": <<tool_name>>}}.
        """
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        tool_names = [ft["function"]["name"] for ft in formatted_tools]
        if tool_choice:
            if isinstance(tool_choice, dict):
                if not any(
                    tool_choice["function"]["name"] == name for name in tool_names
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, str):
                chosen = [
                    f for f in formatted_tools if f["function"]["name"] == tool_choice
                ]
                if not chosen:
                    raise ValueError(
                        f"Tool choice {tool_choice=} was specified, but the only "
                        f"provided tools were {tool_names}."
                    )
            elif isinstance(tool_choice, bool):
                if len(formatted_tools) > 1:
                    raise ValueError(
                        "tool_choice=True can only be specified when a single tool is "
                        f"passed in. Received {len(tools)} tools."
                    )
                tool_choice = formatted_tools[0]
            else:
                raise ValueError(
                    """Unrecognized tool_choice type. Expected dict having format like 
                    this {"type": "function", "function": {"name": <<tool_name>>}}"""
                    f"Received: {tool_choice}"
                )

        kwargs["tool_choice"] = tool_choice
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
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
                must match the OpenAI function-calling spec or be a valid JSON schema
                with top level 'title' and 'description' keys specified.
            include_raw: If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".
            kwargs: Any other args to bind to model, ``self.bind(..., **kwargs)``.

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

        Example: Pydantic schema (include_raw=False):
            .. code-block:: python

                from langchain_community.chat_models import ChatLlamaCpp
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatLlamaCpp(
                    temperature=0.,
                    model_path="./SanctumAI-meta-llama-3-8b-instruct.Q8_0.gguf",
                    n_ctx=10000,
                    n_gpu_layers=4,
                    n_batch=200,
                    max_tokens=512,
                    n_threads=multiprocessing.cpu_count() - 1,
                    repeat_penalty=1.5,
                    top_p=0.5,
                    stop=["<|end_of_text|>", "<|eot_id|>"],
                )
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'
                # )

        Example: Pydantic schema (include_raw=True):
            .. code-block:: python

                from langchain_community.chat_models import ChatLlamaCpp
                from langchain_core.pydantic_v1 import BaseModel

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                llm = ChatLlamaCpp(
                    temperature=0.,
                    model_path="./SanctumAI-meta-llama-3-8b-instruct.Q8_0.gguf",
                    n_ctx=10000,
                    n_gpu_layers=4,
                    n_batch=200,
                    max_tokens=512,
                    n_threads=multiprocessing.cpu_count() - 1,
                    repeat_penalty=1.5,
                    top_p=0.5,
                    stop=["<|end_of_text|>", "<|eot_id|>"],
                )
                structured_llm = llm.with_structured_output(AnswerWithJustification, include_raw=True)

                structured_llm.invoke("What weighs more a pound of bricks or a pound of feathers")
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_Ao02pnFYXD6GN1yzc0uXPsvF', 'function': {'arguments': '{"answer":"They weigh the same.","justification":"Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ."}', 'name': 'AnswerWithJustification'}, 'type': 'function'}]}),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='Both a pound of bricks and a pound of feathers weigh one pound. The weight is the same, but the volume or density of the objects may differ.'),
                #     'parsing_error': None
                # }

        Example: dict schema (include_raw=False):
            .. code-block:: python

                from langchain_community.chat_models import ChatLlamaCpp
                from langchain_core.pydantic_v1 import BaseModel
                from langchain_core.utils.function_calling import convert_to_openai_tool

                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''
                    answer: str
                    justification: str

                dict_schema = convert_to_openai_tool(AnswerWithJustification)
                llm = ChatLlamaCpp(
                    temperature=0.,
                    model_path="./SanctumAI-meta-llama-3-8b-instruct.Q8_0.gguf",
                    n_ctx=10000,
                    n_gpu_layers=4,
                    n_batch=200,
                    max_tokens=512,
                    n_threads=multiprocessing.cpu_count() - 1,
                    repeat_penalty=1.5,
                    top_p=0.5,
                    stop=["<|end_of_text|>", "<|eot_id|>"],
                )
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
        if schema is None:
            raise ValueError(
                "schema must be specified when method is 'function_calling'. "
                "Received None."
            )
        tool_name = convert_to_openai_tool(schema)["function"]["name"]
        llm = self.bind_tools([schema], tool_choice=tool_name)
        if is_pydantic_schema:
            output_parser: OutputParserLike = PydanticToolsParser(
                tools=[cast(Type, schema)], first_tool_only=True
            )
        else:
            output_parser = JsonOutputKeyToolsParser(
                key_name=tool_name, first_tool_only=True
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

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            # The model name allows users to specify custom token counting
            # rules in LLM monitoring applications (e.g., in LangSmith users
            # can provide per token pricing for their model and monitor
            # costs for the given LLM.)
            **{"model_path": self.model_path},
            **self._default_params,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "llama-cpp-python"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling create_chat_completion."""
        params: Dict = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "logprobs": self.logprobs,
            "stop_sequences": self.stop,  # key here is convention among LLM classes
            "repeat_penalty": self.repeat_penalty,
        }
        if self.grammar:
            params["grammar"] = self.grammar
        return params


def _lc_tool_call_to_openai_tool_call(tool_call: ToolCall) -> dict:
    return {
        "type": "function",
        "id": tool_call["id"],
        "function": {
            "name": tool_call["name"],
            "arguments": json.dumps(tool_call["args"]),
        },
    }


def _lc_invalid_tool_call_to_openai_tool_call(
    invalid_tool_call: InvalidToolCall,
) -> dict:
    return {
        "type": "function",
        "id": invalid_tool_call["id"],
        "function": {
            "name": invalid_tool_call["name"],
            "arguments": invalid_tool_call["args"],
        },
    }


def _convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
    """Convert a dictionary to a LangChain message.

    Args:
        _dict: The dictionary.

    Returns:
        The LangChain message.
    """
    role = _dict.get("role")
    name = _dict.get("name")
    id_ = _dict.get("id")
    if role == "user":
        return HumanMessage(content=_dict.get("content", ""), id=id_, name=name)
    elif role == "assistant":
        # Fix for azure
        # Also OpenAI returns None for tool invocations
        content = _dict.get("content", "") or ""
        additional_kwargs: Dict = {}
        if function_call := _dict.get("function_call"):
            additional_kwargs["function_call"] = dict(function_call)
        tool_calls = []
        invalid_tool_calls = []
        if raw_tool_calls := _dict.get("tool_calls"):
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                try:
                    tc = parse_tool_call(raw_tool_call, return_id=True)
                except Exception as e:
                    invalid_tc = make_invalid_tool_call(raw_tool_call, str(e))
                    invalid_tool_calls.append(invalid_tc)
                else:
                    if not tc:
                        continue
                    else:
                        tool_calls.append(tc)
        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
            tool_calls=tool_calls,  # type: ignore[arg-type]
            invalid_tool_calls=invalid_tool_calls,
        )
    elif role == "system":
        return SystemMessage(content=_dict.get("content", ""), name=name, id=id_)
    elif role == "function":
        return FunctionMessage(
            content=_dict.get("content", ""), name=cast(str, _dict.get("name")), id=id_
        )
    elif role == "tool":
        additional_kwargs = {}
        if "name" in _dict:
            additional_kwargs["name"] = _dict["name"]
        return ToolMessage(
            content=_dict.get("content", ""),
            tool_call_id=cast(str, _dict.get("tool_call_id")),
            additional_kwargs=additional_kwargs,
            name=name,
            id=id_,
        )
    else:
        return ChatMessage(
            content=_dict.get("content", ""), role=cast(str, role), id=id_
        )


def _format_message_content(content: Any) -> Any:
    """Format message content."""
    if content and isinstance(content, list):
        # Remove unexpected block types
        formatted_content = []
        for block in content:
            if (
                isinstance(block, dict)
                and "type" in block
                and block["type"] == "tool_use"
            ):
                continue
            else:
                formatted_content.append(block)
    else:
        formatted_content = content

    return formatted_content


def _convert_message_to_dict(message: BaseMessage) -> dict:
    """Convert a LangChain message to a dictionary.

    Args:
        message: The LangChain message.

    Returns:
        The dictionary.
    """
    message_dict: Dict[str, Any] = {
        "content": _format_message_content(message.content),
    }
    if (name := message.name or message.additional_kwargs.get("name")) is not None:
        message_dict["name"] = name

    # populate role and additional message data
    if isinstance(message, ChatMessage):
        message_dict["role"] = message.role
    elif isinstance(message, HumanMessage):
        message_dict["role"] = "user"
    elif isinstance(message, AIMessage):
        message_dict["role"] = "assistant"
        if "function_call" in message.additional_kwargs:
            message_dict["function_call"] = message.additional_kwargs["function_call"]
        if message.tool_calls or message.invalid_tool_calls:
            message_dict["tool_calls"] = [
                _lc_tool_call_to_openai_tool_call(tc) for tc in message.tool_calls
            ] + [
                _lc_invalid_tool_call_to_openai_tool_call(tc)
                for tc in message.invalid_tool_calls
            ]
        elif "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            tool_call_supported_props = {"id", "type", "function"}
            message_dict["tool_calls"] = [
                {k: v for k, v in tool_call.items() if k in tool_call_supported_props}
                for tool_call in message_dict["tool_calls"]
            ]
        else:
            pass
        # If tool calls present, content null value should be None not empty string.
        if "function_call" in message_dict or "tool_calls" in message_dict:
            message_dict["content"] = message_dict["content"] or None
    elif isinstance(message, SystemMessage):
        message_dict["role"] = "system"
    elif isinstance(message, FunctionMessage):
        message_dict["role"] = "function"
    elif isinstance(message, ToolMessage):
        message_dict["role"] = "tool"
        message_dict["tool_call_id"] = message.tool_call_id

        supported_props = {"content", "role", "tool_call_id"}
        message_dict = {k: v for k, v in message_dict.items() if k in supported_props}
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: Type[BaseMessageChunk]
) -> BaseMessageChunk:
    id_ = _dict.get("id")
    role = cast(str, _dict.get("role"))
    content = cast(str, _dict.get("content") or "")
    additional_kwargs: Dict = {}
    if _dict.get("function_call"):
        function_call = dict(_dict["function_call"])
        if "name" in function_call and function_call["name"] is None:
            function_call["name"] = ""
        additional_kwargs["function_call"] = function_call
    tool_call_chunks = []
    if raw_tool_calls := _dict.get("tool_calls"):
        additional_kwargs["tool_calls"] = raw_tool_calls
        for rtc in raw_tool_calls:
            try:
                tool_call = ToolCallChunk(
                    name=rtc["function"].get("name"),
                    args=rtc["function"].get("arguments"),
                    id=rtc.get("id"),
                    index=rtc["index"],
                )
                tool_call_chunks.append(tool_call)
            except KeyError:
                pass

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content, id=id_)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(
            content=content,
            additional_kwargs=additional_kwargs,
            id=id_,
            tool_call_chunks=tool_call_chunks,
        )
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content, id=id_)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"], id=id_)
    elif role == "tool" or default_class == ToolMessageChunk:
        return ToolMessageChunk(
            content=content, tool_call_id=_dict["tool_call_id"], id=id_
        )
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role, id=id_)
    else:
        return default_class(content=content, id=id_)  # type: ignore
