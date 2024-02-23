"""Chat Model Components Derived from ChatModel/NVIDIA"""

from __future__ import annotations

import base64
import io
import logging
import os
import sys
import urllib.parse
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
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

import requests
from langchain.chat_models.base import BaseChatModel
from langchain.output_parsers import PydanticOutputParser
from langchain_core.callbacks.manager import (
    BaseCallbackHandler,
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    ChatMessageChunk,
)
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.output_parsers import (
    JsonOutputParser,
    BaseGenerationOutputParser,
)
from langchain.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
)
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.runnables.config import run_in_executor
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import (
    convert_to_openai_function,
    convert_to_openai_tool,
)

from langchain_nvidia_ai_endpoints import _common as nvidia_ai_endpoints

_DictOrPydanticClass = Union[Dict[str, Any], Type[BaseModel]]
_DictOrPydantic = Union[Dict, BaseModel]

try:
    import PIL.Image

    has_pillow = True
except ImportError:
    has_pillow = False

logger = logging.getLogger(__name__)


def _is_url(s: str) -> bool:
    try:
        result = urllib.parse.urlparse(s)
        return all([result.scheme, result.netloc])
    except Exception as e:
        logger.debug(f"Unable to parse URL: {e}")
        return False


def _resize_image(img_data: bytes, max_dim: int = 1024) -> str:
    if not has_pillow:
        print(  # noqa: T201
            "Pillow is required to resize images down to reasonable scale."
            " Please install it using `pip install pillow`."
            " For now, not resizing; may cause NVIDIA API to fail."
        )
        return base64.b64encode(img_data).decode("utf-8")
    image = PIL.Image.open(io.BytesIO(img_data))
    max_dim_size = max(image.size)
    aspect_ratio = max_dim / max_dim_size
    new_h = int(image.size[1] * aspect_ratio)
    new_w = int(image.size[0] * aspect_ratio)
    resized_image = image.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    output_buffer = io.BytesIO()
    resized_image.save(output_buffer, format="JPEG")
    output_buffer.seek(0)
    resized_b64_string = base64.b64encode(output_buffer.read()).decode("utf-8")
    return resized_b64_string


def _url_to_b64_string(image_source: str) -> str:
    b64_template = "data:image/png;base64,{b64_string}"
    try:
        if _is_url(image_source):
            response = requests.get(image_source)
            response.raise_for_status()
            encoded = base64.b64encode(response.content).decode("utf-8")
            if sys.getsizeof(encoded) > 200000:
                ## (VK) Temporary fix. NVIDIA API has a limit of 250KB for the input.
                encoded = _resize_image(response.content)
            return b64_template.format(b64_string=encoded)
        elif image_source.startswith("data:image"):
            return image_source
        elif os.path.exists(image_source):
            with open(image_source, "rb") as f:
                encoded = base64.b64encode(f.read()).decode("utf-8")
                return b64_template.format(b64_string=encoded)
        else:
            raise ValueError(
                "The provided string is not a valid URL, base64, or file path."
            )
    except Exception as e:
        raise ValueError(f"Unable to process the provided image source: {e}")


class ChatNVIDIA(nvidia_ai_endpoints._NVIDIAClient, BaseChatModel):
    """NVIDIA chat model.

    Example:
        .. code-block:: python

            from langchain_nvidia_ai_endpoints import ChatNVIDIA


            model = ChatNVIDIA(model="llama2_13b")
            response = model.invoke("Hello")
    """

    temperature: Optional[float] = Field(description="Sampling temperature in [0, 1]")
    max_tokens: Optional[int] = Field(description="Maximum # of tokens to generate")
    top_p: Optional[float] = Field(description="Top-p for distribution sampling")
    seed: Optional[int] = Field(description="The seed for deterministic results")
    bad: Optional[Sequence[str]] = Field(description="Bad words to avoid (cased)")
    stop: Optional[Sequence[str]] = Field(description="Stop words (cased)")
    labels: Optional[Dict[str, float]] = Field(description="Steering parameters")
    streaming: bool = Field(True)

    @property
    def _llm_type(self) -> str:
        """Return type of NVIDIA AI Foundation Model Interface."""
        return "chat-nvidia-ai-playground"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        responses = self._call(messages, stop=stop, run_manager=run_manager, **kwargs)
        self._set_callback_outputs(responses, run_manager)
        outputs = self.custom_postprocess(responses)
        message = AIMessage(content=outputs)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation], llm_output=responses)

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        return await run_in_executor(
            None,
            self._generate,
            messages,
            stop=stop,
            run_manager=run_manager.get_sync() if run_manager else None,
            **kwargs,
        )

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> dict:
        """Invoke on a single list of chat messages."""
        inputs = self.custom_preprocess(messages)
        responses = self.get_generation(inputs=inputs, stop=stop, **kwargs)
        return responses

    def _get_filled_chunk(
        self, text: str, role: Optional[str] = "assistant"
    ) -> ChatGenerationChunk:
        """Fill the generation chunk."""
        return ChatGenerationChunk(message=ChatMessageChunk(content=text, role=role))

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Allows streaming to model!"""
        inputs = self.custom_preprocess(messages)
        for response in self.get_stream(inputs=inputs, stop=stop, **kwargs):
            self._set_callback_outputs(response, run_manager)
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[Sequence[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        inputs = self.custom_preprocess(messages)
        async for response in self.get_astream(inputs=inputs, stop=stop, **kwargs):
            self._set_callback_outputs(response, run_manager)
            chunk = self._get_filled_chunk(self.custom_postprocess(response))
            yield chunk
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
    
    def _set_callback_outputs(self, results: dict, run_manager: BaseCallbackHandler) -> None:
        results.update({"model_name" : self.model})
        for cb in run_manager.handlers:
            if hasattr(cb, "llm_output"):
                cb.llm_output = results

    def custom_preprocess(
        self, msg_list: Sequence[BaseMessage]
    ) -> List[Dict[str, str]]:
        return [self.preprocess_msg(m) for m in msg_list]

    def _process_content(self, content: Union[str, List[Union[dict, str]]]) -> str:
        if isinstance(content, str):
            return content
        string_array: list = []

        for part in content:
            if isinstance(part, str):
                string_array.append(part)
            elif isinstance(part, Mapping):
                # OpenAI Format
                if "type" in part:
                    if part["type"] == "text":
                        string_array.append(str(part["text"]))
                    elif part["type"] == "image_url":
                        img_url = part["image_url"]
                        if isinstance(img_url, dict):
                            if "url" not in img_url:
                                raise ValueError(
                                    f"Unrecognized message image format: {img_url}"
                                )
                            img_url = img_url["url"]
                        b64_string = _url_to_b64_string(img_url)
                        string_array.append(f'<img src="{b64_string}" />')
                    else:
                        raise ValueError(
                            f"Unrecognized message part type: {part['type']}"
                        )
                else:
                    raise ValueError(f"Unrecognized message part format: {part}")
        return "".join(string_array)

    def preprocess_msg(self, msg: BaseMessage) -> Dict[str, str]:
        if isinstance(msg, BaseMessage):
            role_convert = {"ai": "assistant", "human": "user"}
            if isinstance(msg, ChatMessage):
                role = msg.role
            else:
                role = msg.type
            role = role_convert.get(role, role)
            content = self._process_content(msg.content)
            return {"role": role, "content": content}
        raise ValueError(f"Invalid message: {repr(msg)} of type {type(msg)}")

    def custom_postprocess(self, msg: dict) -> str:
        if "content" in msg:
            return msg["content"]
        elif "b64_json" in msg:
            return msg["b64_json"]
        return str(msg)

    ######################################################################################
    ## Core client-side interfaces

    def get_generation(
        self,
        inputs: Sequence[Dict],
        **kwargs: Any,
    ) -> dict:
        """Call to client generate method with call scope"""
        stop = kwargs.get("stop", None)
        payload = self.get_payload(inputs=inputs, stream=False, **kwargs)
        out = self.client.get_req_generation(self.model, stop=stop, payload=payload)
        return out

    def get_stream(
        self,
        inputs: Sequence[Dict],
        **kwargs: Any,
    ) -> Iterator:
        """Call to client stream method with call scope"""
        stop = kwargs.get("stop", None)
        payload = self.get_payload(inputs=inputs, stream=True, **kwargs)
        return self.client.get_req_stream(self.model, stop=stop, payload=payload)

    def get_astream(
        self,
        inputs: Sequence[Dict],
        **kwargs: Any,
    ) -> AsyncIterator:
        """Call to client astream methods with call scope"""
        stop = kwargs.get("stop", None)
        payload = self.get_payload(inputs=inputs, stream=True, **kwargs)
        return self.client.get_req_astream(self.model, stop=stop, payload=payload)

    def get_payload(self, inputs: Sequence[Dict], **kwargs: Any) -> dict:
        """Generates payload for the _NVIDIAClient API to send to service."""
        attr_kwargs = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "seed": self.seed,
            "bad": self.bad,
            "stop": self.stop,
            "labels": self.labels,
        }
        attr_kwargs = {k: v for k, v in attr_kwargs.items() if v is not None}
        new_kwargs = {**attr_kwargs, **kwargs}
        return self.prep_payload(inputs=inputs, **new_kwargs)

    def prep_payload(self, inputs: Sequence[Dict], **kwargs: Any) -> dict:
        """Prepares a message or list of messages for the payload"""
        messages = [self.prep_msg(m) for m in inputs]
        if kwargs.get("labels"):
            # (WFH) Labels are currently (?) always passed as an assistant
            # suffix message, but this API seems less stable.
            messages += [{"labels": kwargs.pop("labels"), "role": "assistant"}]
        if kwargs.get("stop") is None:
            kwargs.pop("stop")
        return {"messages": messages, **kwargs}

    def prep_msg(self, msg: Union[str, dict, BaseMessage]) -> dict:
        """Helper Method: Ensures a message is a dictionary with a role and content."""
        if isinstance(msg, str):
            # (WFH) this shouldn't ever be reached but leaving this here bcs
            # it's a Chesterton's fence I'm unwilling to touch
            return dict(role="user", content=msg)
        if isinstance(msg, dict):
            if msg.get("content", None) is None:
                raise ValueError(f"Message {msg} has no content")
            return msg
        raise ValueError(f"Unknown message received: {msg} of type {type(msg)}")

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        *,
        tool_choice: Optional[Union[dict, str, Literal["auto", "none"], bool]] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model.

        Assumes model is compatible with OpenAI tool-calling API.

        Args:
            tools: A list of tool definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, callable, or BaseTool. Pydantic
                models, callables, and BaseTools will be automatically converted to
                their schema dictionary representation.
            tool_choice: Which tool to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any), or a dict of the form:
                {"type": "function", "function": {"name": <<tool_name>>}}.
            **kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.

        See ChatOpenAI.bind_tools. This served as preemptive support
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        if tool_choice is not None and tool_choice:
            if len(formatted_tools) != 1:
                raise ValueError(
                    "When specifying `tool_choice`, you must provide exactly one "
                    f"tool. Received {len(formatted_tools)} tools."
                )
            if isinstance(tool_choice, str):
                if tool_choice not in ("auto", "none"):
                    tool_choice = {
                        "type": "function",
                        "function": {"name": tool_choice},
                    }
            elif isinstance(tool_choice, bool):
                tool_choice = formatted_tools[0]
            elif isinstance(tool_choice, dict):
                if (
                    formatted_tools[0]["function"]["name"]
                    != tool_choice["function"]["name"]
                ):
                    raise ValueError(
                        f"Tool choice {tool_choice} was specified, but the only "
                        f"provided tool was {formatted_tools[0]['function']['name']}."
                    )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool or dict. "
                    f"Received: {tool_choice}"
                )
            kwargs["tool_choice"] = tool_choice
        return super().bind(tools=formatted_tools, **kwargs)


    def bind_functions(
        self,
        functions: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable]],
        function_call: Optional[str] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind functions (and other objects) to this chat model.

        Args:
            functions: A list of function definitions to bind to this chat model.
                Can be  a dictionary, pydantic model, or callable. Pydantic
                models and callables will be automatically converted to
                their schema dictionary representation.
            function_call: Which function to require the model to call.
                Must be the name of the single provided function or
                "auto" to automatically determine which function to call
                (if any).
            kwargs: Any additional parameters to pass to the
                :class:`~langchain.runnable.Runnable` constructor.

        See ChatOpenAI.bind_functions. This served as preemptive support
        """
        formatted_functions = [convert_to_openai_function(fn) for fn in functions]
        if function_call is not None:
            if len(formatted_functions) != 1:
                raise ValueError(
                    "When specifying `function_call`, you must provide exactly one "
                    "function."
                )
            if formatted_functions[0]["name"] != function_call:
                raise ValueError(
                    f"Function call {function_call} was specified, but the only "
                    f"provided function was {formatted_functions[0]['name']}."
                )
            function_call_ = {"name": function_call}
            kwargs = {**kwargs, "function_call": function_call_}
        return super().bind(
            functions=formatted_functions,
            **kwargs,
        )

    def with_structured_output(
        self,
        schema: _DictOrPydanticClass,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        return_type: Literal["parsed", "all"] = "parsed",
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema as a dict or a Pydantic class. If a Pydantic class
                then the model output will be an object of that class. If a dict then
                the model output will be a dict. With a Pydantic class the returned
                attributes will be validated, whereas with a dict they will not be. If
                `method` is "function_calling" and `schema` is a dict, then the dict
                must match the OpenAI function-calling spec.
            method: The method for steering model generation, either "function_calling"
                or "json_mode". If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" then OpenAI's JSON mode will be
                used.
            return_type: The wrapped model's return type, either "parsed" or "all". If
                "parsed" then only the parsed structured output is returned. If an
                error occurs during model output parsing it will be raised. If "all"
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes any ChatModel input and returns as output:

                If return_type == "all" then a dict with keys:
                    raw: BaseMessage
                    parsed: Optional[_DictOrPydantic]
                    parsing_error: Optional[BaseException]

                If return_type == "parsed" then just _DictOrPydantic is returned,
                where _DictOrPydantic depends on the schema:

                If schema is a Pydantic class then _DictOrPydantic is the Pydantic
                    class.

                If schema is a dict then _DictOrPydantic is a dict.

        Example: See ChatOpenAI.with_structured_output
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = isinstance(schema, type) and issubclass(schema, BaseModel)
        if method == "function_calling":
            llm = self.bind_tools([schema], tool_choice=True)
            if is_pydantic_schema:
                output_parser: BaseGenerationOutputParser = PydanticToolsParser(
                    tools=[schema], first_tool_only=True
                )
            else:
                key_name = convert_to_openai_tool(schema)["function"]["name"]
                output_parser = JsonOutputKeyToolsParser(
                    key_name=key_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_format'. Received: '{method}'"
            )

        if return_type == "parsed":
            return llm | output_parser
        elif return_type == "all":
            parser_assign = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | output_parser, parsing_error=lambda _: None
            )
            parser_none = RunnablePassthrough.assign(parsed=lambda _: None)
            parser_with_fallback = parser_assign.with_fallbacks(
                [parser_none], exception_key="parsing_error"
            )
            return RunnableMap(raw=llm) | parser_with_fallback
        else:
            raise ValueError(
                f"Unrecognized return_type argument. Expected one of 'parsed' or "
                f"'all'. Received: '{return_type}'"
            )
