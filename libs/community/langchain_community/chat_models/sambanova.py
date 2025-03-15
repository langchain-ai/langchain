import json
from operator import itemgetter
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import requests
from langchain_core._api.deprecation import deprecated
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
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
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.output_parsers.openai_tools import (
    JsonOutputKeyToolsParser,
    PydanticToolsParser,
    make_invalid_tool_call,
    parse_tool_call,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.utils.pydantic import is_basemodel_subclass
from pydantic import BaseModel, Field, SecretStr
from requests import Response


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """
    convert a BaseMessage to a dictionary with Role / content

    Args:
        message: BaseMessage

    Returns:
        messages_dict:  role / content dict
    """
    message_dict: Dict[str, Any] = {}
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
        if "tool_calls" in message.additional_kwargs:
            message_dict["tool_calls"] = message.additional_kwargs["tool_calls"]
            if message_dict["content"] == "":
                message_dict["content"] = None
    elif isinstance(message, ToolMessage):
        message_dict = {
            "role": "tool",
            "content": message.content,
            "tool_call_id": message.tool_call_id,
        }
    else:
        raise TypeError(f"Got unknown type {message}")
    return message_dict


def _create_message_dicts(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
    """
    Convert a list of BaseMessages to a list of dictionaries with Role / content

    Args:
        messages: list of BaseMessages

    Returns:
        messages_dicts:  list of role / content dicts
    """
    message_dicts = [_convert_message_to_dict(m) for m in messages]
    return message_dicts


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and is_basemodel_subclass(obj)


@deprecated(
    since="0.3.16",
    removal="1.0",
    alternative_import="langchain_sambanova.ChatSambaNovaCloud",
)
class ChatSambaNovaCloud(BaseChatModel):
    """
    SambaNova Cloud chat model.

    Setup:
        To use, you should have the environment variables:
        `SAMBANOVA_URL` set with your SambaNova Cloud URL.
        `SAMBANOVA_API_KEY` set with your SambaNova Cloud API Key.
        http://cloud.sambanova.ai/
        Example:
        .. code-block:: python
            ChatSambaNovaCloud(
                sambanova_url = SambaNova cloud endpoint URL,
                sambanova_api_key = set with your SambaNova cloud API key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                stream_options = include usage to get generation metrics
            )

    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct.
        streaming: bool
            Whether to use streaming handler when using non streaming methods
        max_tokens: int
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        top_k: int
            model top k
        stream_options: dict
            stream options, include usage to get generation metrics

    Key init args — client params:
        sambanova_url: str
            SambaNova Cloud Url
        sambanova_api_key: str
            SambaNova Cloud api key

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatSambaNovaCloud

            chat = ChatSambaNovaCloud(
                sambanova_url = SambaNova cloud endpoint URL,
                sambanova_api_key = set with your SambaNova cloud API key,
                model = model name,
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                stream_options = include usage to get generation metrics
            )

    Invoke:
        .. code-block:: python

            messages = [
                SystemMessage(content="your are an AI assistant."),
                HumanMessage(content="tell me a joke."),
            ]
            response = chat.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            response = chat.ainvoke(messages)
            await response

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ...,
                    description="The city and state, e.g. Los Angeles, CA"
                )

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Should I bring my umbrella today in LA?")
            ai_msg.tool_calls

        .. code-block:: none

            [
                {
                    'name': 'GetWeather',
                    'args': {'location': 'Los Angeles, CA'},
                    'id': 'call_adf61180ea2b4d228a'
                }
            ]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            structured_model = llm.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup="Why did the cat join a band?",
            punchline="Because it wanted to be the purr-cussionist!")

        See `ChatSambanovaCloud.with_structured_output()` for more.

    Token usage:
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.response_metadata["usage"]["prompt_tokens"]
            print(response.response_metadata["usage"]["total_tokens"]

    Response metadata
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.response_metadata)

    """

    sambanova_url: str = Field(default="")
    """SambaNova Cloud Url"""

    sambanova_api_key: SecretStr = Field(default=SecretStr(""))
    """SambaNova Cloud api key"""

    model: str = Field(default="Meta-Llama-3.1-8B-Instruct")
    """The name of the model"""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: float = Field(default=0.7)
    """model temperature"""

    top_p: Optional[float] = Field(default=None)
    """model top p"""

    top_k: Optional[int] = Field(default=None)
    """model top k"""

    stream_options: Dict[str, Any] = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""

    additional_headers: Dict[str, Any] = Field(default={})
    """Additional headers to sent in request"""

    class Config:
        populate_by_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"sambanova_api_key": "sambanova_api_key"}

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model": self.model,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "stream_options": self.stream_options,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sambanovacloud-chatmodel"

    def __init__(self, **kwargs: Any) -> None:
        """init and validate environment variables"""
        kwargs["sambanova_url"] = get_from_dict_or_env(
            kwargs,
            "sambanova_url",
            "SAMBANOVA_URL",
            default="https://api.sambanova.ai/v1/chat/completions",
        )
        kwargs["sambanova_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "sambanova_api_key", "SAMBANOVA_API_KEY")
        )
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[Any], Callable[..., Any], BaseTool]],
        *,
        tool_choice: Optional[Union[Dict[str, Any], bool, str]] = None,
        parallel_tool_calls: Optional[bool] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model

        tool_choice: does not currently support "any", choice like
        should be one of ["auto", "none", "required"]
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "required"):
                    tool_choice = "auto"
            elif isinstance(tool_choice, bool):
                if tool_choice:
                    tool_choice = "required"
            elif isinstance(tool_choice, dict):
                raise ValueError(
                    "tool_choice must be one of ['auto', 'none', 'required']"
                )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool"
                    f"Received: {tool_choice}"
                )
        else:
            tool_choice = "auto"
        kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict[str, Any], BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class,
                    - or a Pydantic.BaseModel class.
                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method:
                The method for steering model generation, either "function_calling"
                "json_mode" or "json_schema".
                If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" or "json_schema" then OpenAI's
                JSON mode will be used.
                Note that if using "json_mode" or "json_schema" then you must include instructions
                for formatting the output into the desired schema into the model call.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If `include_raw` is False and `schema` is a Pydantic class, Runnable outputs
            an instance of `schema` (i.e., a Pydantic object).

            Otherwise, if `include_raw` is False then Runnable outputs a dict.

            If `include_raw` is True, then Runnable outputs a dict with keys:
                - `"raw"`: BaseMessage
                - `"parsed"`: None if there was a parsing error, otherwise the type depends on the `schema` as described above.
                - `"parsing_error"`: Optional[BaseException]

        Example: schema=Pydantic class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_community.chat_models import ChatSambaNovaCloud
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str = Field(
                        description="A justification for the answer."
                    )


                llm = ChatSambaNovaCloud(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_community.chat_models import ChatSambaNovaCloud
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatSambaNovaCloud(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"answer": "They weigh the same.", "justification": "A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount."}', 'name': 'AnswerWithJustification'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'usage': {'acceptance_rate': 5, 'completion_tokens': 53, 'completion_tokens_after_first_per_sec': 343.7964936837758, 'completion_tokens_after_first_per_sec_first_ten': 439.1205661878638, 'completion_tokens_per_sec': 162.8511306784833, 'end_time': 1731527851.0698032, 'is_last_response': True, 'prompt_tokens': 213, 'start_time': 1731527850.7137961, 'time_to_first_token': 0.20475482940673828, 'total_latency': 0.32545061111450196, 'total_tokens': 266, 'total_tokens_per_sec': 817.3283162354066}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731527850}, id='95667eaf-447f-4b53-bb6e-b6e1094ded88', tool_calls=[{'name': 'AnswerWithJustification', 'args': {'answer': 'They weigh the same.', 'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'tool_call'}]),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'),
                #     'parsing_error': None
                # }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_community.chat_models import ChatSambaNovaCloud


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[
                        Optional[str], None, "A justification for the answer."
                    ]


                llm = ChatSambaNovaCloud(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_community.chat_models import ChatSambaNovaCloud

                oai_schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
                }

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(oai_schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=Pydantic class, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_community.chat_models import ChatSambaNovaCloud
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_community.chat_models import ChatSambaNovaCloud

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 4.722222222222222, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 357.1315485254867, 'completion_tokens_after_first_per_sec_first_ten': 416.83279609305305, 'completion_tokens_per_sec': 240.92819585198137, 'end_time': 1731528164.8474727, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528164.4906917, 'time_to_first_token': 0.13837409019470215, 'total_latency': 0.3278985247892492, 'total_tokens': 149, 'total_tokens_per_sec': 454.4088757208256}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528164}, id='15261eaf-8a25-42ef-8ed5-f63d8bf5b1b0'),
                #     'parsed': {
                #         'answer': 'They are the same weight',
                #         'justification': 'A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'},
                #     },
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_schema", include_raw=True:
            .. code-block::

                from langchain_community.chat_models import ChatSambaNovaCloud

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaNovaCloud(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, method="json_schema", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }
        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "`schema` must be specified when method is `function_calling`. "
                    "Received None."
                )
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools([schema], tool_choice=tool_name)
            if is_pydantic_schema:
                output_parser: OutputParserLike[Any] = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,  # type: ignore[list-item]
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self
            # TODO bind response format when json mode available by API
            # llm = self.bind(response_format={"type": "json_object"})
            if is_pydantic_schema:
                schema = cast(Type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
            else:
                output_parser = JsonOutputParser()

        elif method == "json_schema":
            if schema is None:
                raise ValueError(
                    "`schema` must be specified when method is not `json_mode`. "
                    "Received None."
                )
            llm = self
            # TODO bind response format when json schema available by API,
            # update example
            # llm = self.bind(
            #   response_format={"type": "json_object", "json_schema": schema}
            # )
            if is_pydantic_schema:
                schema = cast(Type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
            else:
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of `function_calling` or "
                f"`json_mode`. Received: `{method}`"
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

    def _handle_request(
        self,
        messages_dicts: List[Dict[str, Any]],
        stop: Optional[List[str]] = None,
        streaming: bool = False,
        **kwargs: Any,
    ) -> Response:
        """
        Performs a post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens
            streaming: wether to do a streaming call

        Returns:
            An iterator of response dicts.
        """
        if streaming:
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stream": True,
                "stream_options": self.stream_options,
                **kwargs,
            }
        else:
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                **kwargs,
            }
        http_session = requests.Session()
        response = http_session.post(
            self.sambanova_url,
            headers={
                "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                **self.additional_headers,
            },
            json=data,
            stream=streaming,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}.",
                f"{response.text}.",
            )
        return response

    def _process_response(self, response: Response) -> AIMessage:
        """
        Process a non streaming response from the api

        Args:
            response: A request Response object

        Returns
            generation: an AIMessage with model generation
        """
        try:
            response_dict = response.json()
            if response_dict.get("error"):
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}.",
                    f"{response_dict}.",
                )
        except Exception as e:
            raise RuntimeError(
                f"Sambanova /complete call failed couldn't get JSON response {e}"
                f"response: {response.text}"
            )
        content = response_dict["choices"][0]["message"].get("content", "")
        if content is None:
            content = ""
        additional_kwargs: Dict[str, Any] = {}
        tool_calls = []
        invalid_tool_calls = []
        raw_tool_calls = response_dict["choices"][0]["message"].get("tool_calls")
        if raw_tool_calls:
            additional_kwargs["tool_calls"] = raw_tool_calls
            for raw_tool_call in raw_tool_calls:
                if isinstance(raw_tool_call["function"]["arguments"], dict):
                    raw_tool_call["function"]["arguments"] = json.dumps(
                        raw_tool_call["function"].get("arguments", {})
                    )
                try:
                    tool_calls.append(parse_tool_call(raw_tool_call, return_id=True))
                except Exception as e:
                    invalid_tool_calls.append(
                        make_invalid_tool_call(raw_tool_call, str(e))
                    )
        message = AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata={
                "finish_reason": response_dict["choices"][0]["finish_reason"],
                "usage": response_dict.get("usage"),
                "model_name": response_dict["model"],
                "system_fingerprint": response_dict["system_fingerprint"],
                "created": response_dict["created"],
            },
            id=response_dict["id"],
        )
        return message

    def _process_stream_response(
        self, response: Response
    ) -> Iterator[BaseMessageChunk]:
        """
        Process a streaming response from the api

        Args:
            response: An iterable request Response object

        Yields:
            generation: an AIMessageChunk with model partial generation
        """
        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )

        client = sseclient.SSEClient(response)

        for event in client.events():
            if event.event == "error_event":
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{response.status_code}."
                    f"{event.data}."
                )

            try:
                # check if the response is a final event
                # in that case event data response is '[DONE]'
                if event.data != "[DONE]":
                    if isinstance(event.data, str):
                        data = json.loads(event.data)
                    else:
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                    if data.get("error"):
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{response.status_code}."
                            f"{event.data}."
                        )
                    if len(data["choices"]) > 0:
                        finish_reason = data["choices"][0].get("finish_reason")
                        content = data["choices"][0]["delta"]["content"]
                        id = data["id"]
                        chunk = AIMessageChunk(
                            content=content, id=id, additional_kwargs={}
                        )
                    else:
                        content = ""
                        id = data["id"]
                        metadata = {
                            "finish_reason": finish_reason,
                            "usage": data.get("usage"),
                            "model_name": data["model"],
                            "system_fingerprint": data["system_fingerprint"],
                            "created": data["created"],
                        }
                        chunk = AIMessageChunk(
                            content=content,
                            id=id,
                            response_metadata=metadata,
                            additional_kwargs={},
                        )
                    yield chunk

            except Exception as e:
                raise RuntimeError(
                    f"Error getting content chunk raw streamed response: {e}"
                    f"data: {event.data}"
                )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Call SambaNovaCloud models.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.

        Returns:
            result: ChatResult with model generation
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
        messages_dicts = _create_message_dicts(messages)
        response = self._handle_request(messages_dicts, stop, streaming=False, **kwargs)
        message = self._process_response(response)
        generation = ChatGeneration(
            message=message,
            generation_info={
                "finish_reason": message.response_metadata["finish_reason"]
            },
        )
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the SambaNovaCloud chat model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.

        Yields:
            chunk: ChatGenerationChunk with model partial generation
        """
        messages_dicts = _create_message_dicts(messages)
        response = self._handle_request(messages_dicts, stop, streaming=True, **kwargs)
        for ai_message_chunk in self._process_stream_response(response):
            chunk = ChatGenerationChunk(message=ai_message_chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


@deprecated(
    since="0.3.16",
    removal="1.0",
    alternative_import="langchain_sambanova.ChatSambaStudio",
)
class ChatSambaStudio(BaseChatModel):
    """
    SambaStudio chat model.

    Setup:
        To use, you should have the environment variables:
        `SAMBASTUDIO_URL` set with your SambaStudio deployed endpoint URL.
        `SAMBASTUDIO_API_KEY` set with your SambaStudio deployed endpoint Key.
        https://docs.sambanova.ai/sambastudio/latest/index.html
        Example:

        .. code-block:: python

            ChatSambaStudio(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model or expert name (set for Bundle endpoints),
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                do_sample = wether to do sample
                process_prompt = wether to process prompt
                    (set for Bundle generic v1 and v2 endpoints)
                stream_options = include usage to get generation metrics
                special_tokens = start, start_role, end_role, end special tokens
                    (set for Bundle generic v1 and v2 endpoints when process prompt
                     set to false or for StandAlone v1 and v2 endpoints)
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct-4096
            (set for Bundle endpoints).
        streaming: bool
            Whether to use streaming
        max_tokens: inthandler when using non streaming methods
            max tokens to generate
        temperature: float
            model temperature
        top_p: float
            model top p
        top_k: int
            model top k
        do_sample: bool
            wether to do sample
        process_prompt:
            wether to process prompt (set for Bundle generic v1 and v2 endpoints)
        stream_options: dict
            stream options, include usage to get generation metrics
        special_tokens: dict
            start, start_role, end_role and end special tokens
            (set for Bundle generic v1 and v2 endpoints when process prompt set to false
             or for StandAlone v1 and v2 endpoints) default to llama3 special tokens
        model_kwargs: dict
            Extra Key word arguments to pass to the model.

    Key init args — client params:
        sambastudio_url: str
            SambaStudio endpoint Url
        sambastudio_api_key: str
            SambaStudio endpoint api key

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatSambaStudio

            chat = ChatSambaStudio=(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model or expert name (set for Bundle endpoints),
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                do_sample = wether to do sample
                process_prompt = wether to process prompt
                    (set for Bundle generic v1 and v2 endpoints)
                stream_options = include usage to get generation metrics
                special_tokens = start, start_role, end_role, and special tokens
                    (set for Bundle generic v1 and v2 endpoints when process prompt
                     set to false or for StandAlone v1 and v2 endpoints)
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Invoke:
        .. code-block:: python

            messages = [
                SystemMessage(content="your are an AI assistant."),
                HumanMessage(content="tell me a joke."),
            ]
            response = chat.invoke(messages)

    Stream:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            response = chat.ainvoke(messages)
            await response

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ...,
                    description="The city and state, e.g. Los Angeles, CA"
                )

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke("Should I bring my umbrella today in LA?")
            ai_msg.tool_calls

        .. code-block:: python

            [
                {
                    'name': 'GetWeather',
                    'args': {'location': 'Los Angeles, CA'},
                    'id': 'call_adf61180ea2b4d228a'
                }
            ]

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field

            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            structured_model = llm.with_structured_output(Joke)
            structured_model.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(setup="Why did the cat join a band?",
            punchline="Because it wanted to be the purr-cussionist!")

        See `ChatSambaStudio.with_structured_output()` for more.

    Token usage:
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.response_metadata["usage"]["prompt_tokens"]
            print(response.response_metadata["usage"]["total_tokens"]

    Response metadata
        .. code-block:: python

            response = chat.invoke(messages)
            print(response.response_metadata)
    """

    sambastudio_url: str = Field(default="")
    """SambaStudio Url"""

    sambastudio_api_key: SecretStr = Field(default=SecretStr(""))
    """SambaStudio api key"""

    base_url: str = Field(default="", exclude=True)
    """SambaStudio non streaming Url"""

    streaming_url: str = Field(default="", exclude=True)
    """SambaStudio streaming Url"""

    model: Optional[str] = Field(default=None)
    """The name of the model or expert to use (for Bundle endpoints)"""

    streaming: bool = Field(default=False)
    """Whether to use streaming handler when using non streaming methods"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: Optional[float] = Field(default=0.7)
    """model temperature"""

    top_p: Optional[float] = Field(default=None)
    """model top p"""

    top_k: Optional[int] = Field(default=None)
    """model top k"""

    do_sample: Optional[bool] = Field(default=None)
    """whether to do sampling"""

    process_prompt: Optional[bool] = Field(default=True)
    """whether process prompt (for Bundle generic v1 and v2 endpoints)"""

    stream_options: Dict[str, Any] = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""

    special_tokens: Dict[str, Any] = Field(
        default={
            "start": "<|begin_of_text|>",
            "start_role": "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>",
            "end_role": "<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>\n",
        }
    )
    """start, start_role, end_role and end special tokens 
    (set for Bundle generic v1 and v2 endpoints when process prompt set to false 
     or for StandAlone v1 and v2 endpoints) 
    default to llama3 special tokens"""

    model_kwargs: Optional[Dict[str, Any]] = None
    """Key word arguments to pass to the model."""

    additional_headers: Dict[str, Any] = Field(default={})
    """Additional headers to send in request"""

    class Config:
        populate_by_name = True

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return False

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "sambastudio_url": "sambastudio_url",
            "sambastudio_api_key": "sambastudio_api_key",
        }

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters.

        This information is used by the LangChain callback system, which
        is used for tracing purposes make it possible to monitor LLMs.
        """
        return {
            "model": self.model,
            "streaming": self.streaming,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "do_sample": self.do_sample,
            "process_prompt": self.process_prompt,
            "stream_options": self.stream_options,
            "special_tokens": self.special_tokens,
            "model_kwargs": self.model_kwargs,
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sambastudio-chatmodel"

    def __init__(self, **kwargs: Any) -> None:
        """init and validate environment variables"""
        kwargs["sambastudio_url"] = get_from_dict_or_env(
            kwargs, "sambastudio_url", "SAMBASTUDIO_URL"
        )

        kwargs["sambastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(kwargs, "sambastudio_api_key", "SAMBASTUDIO_API_KEY")
        )
        kwargs["base_url"], kwargs["streaming_url"] = self._get_sambastudio_urls(
            kwargs["sambastudio_url"]
        )
        super().__init__(**kwargs)

    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[Any], Callable[..., Any], BaseTool]],
        *,
        tool_choice: Optional[Union[Dict[str, Any], bool, str]] = None,
        parallel_tool_calls: Optional[bool] = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tool-like objects to this chat model

        tool_choice: does not currently support "any", choice like
        should be one of ["auto", "none", "required"]
        """

        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]

        if tool_choice:
            if isinstance(tool_choice, str):
                # tool_choice is a tool/function name
                if tool_choice not in ("auto", "none", "required"):
                    tool_choice = "auto"
            elif isinstance(tool_choice, bool):
                if tool_choice:
                    tool_choice = "required"
            elif isinstance(tool_choice, dict):
                raise ValueError(
                    "tool_choice must be one of ['auto', 'none', 'required']"
                )
            else:
                raise ValueError(
                    f"Unrecognized tool_choice type. Expected str, bool"
                    f"Received: {tool_choice}"
                )
        else:
            tool_choice = "auto"
        kwargs["tool_choice"] = tool_choice
        kwargs["parallel_tool_calls"] = parallel_tool_calls
        return super().bind(tools=formatted_tools, **kwargs)

    def with_structured_output(
        self,
        schema: Optional[Union[Dict[str, Any], Type[BaseModel]]] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict[str, Any], BaseModel]]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema:
                The output schema. Can be passed in as:
                    - an OpenAI function/tool schema,
                    - a JSON Schema,
                    - a TypedDict class,
                    - or a Pydantic class.
                If `schema` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method:
                The method for steering model generation, either "function_calling"
                "json_mode" or "json_schema".
                If "function_calling" then the schema will be converted
                to an OpenAI function and the returned model will make use of the
                function-calling API. If "json_mode" or "json_schema" then OpenAI's
                JSON mode will be used.
                Note that if using "json_mode" or "json_schema" then you must include instructions
                for formatting the output into the desired schema into the model call.

            include_raw:
                If False then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If True
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys "raw", "parsed", and "parsing_error".

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If `include_raw` is False and `schema` is a Pydantic class, Runnable outputs
            an instance of `schema` (i.e., a Pydantic object).

            Otherwise, if `include_raw` is False then Runnable outputs a dict.

            If `include_raw` is True, then Runnable outputs a dict with keys:
                - `"raw"`: BaseMessage
                - `"parsed"`: None if there was a parsing error, otherwise the type depends on the `schema` as described above.
                - `"parsing_error"`: Optional[BaseException]

        Example: schema=Pydantic class, method="function_calling", include_raw=False:
            .. code-block:: python

                from typing import Optional

                from langchain_community.chat_models import ChatSambaStudio
                from pydantic import BaseModel, Field


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str = Field(
                        description="A justification for the answer."
                    )


                llm = ChatSambaStudio(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )

                # -> AnswerWithJustification(
                #     answer='They weigh the same',
                #     justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same.'
                # )

        Example: schema=Pydantic class, method="function_calling", include_raw=True:
            .. code-block:: python

                from langchain_community.chat_models import ChatSambaStudio
                from pydantic import BaseModel


                class AnswerWithJustification(BaseModel):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: str


                llm = ChatSambaStudio(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification, include_raw=True
                )

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'raw': AIMessage(content='', additional_kwargs={'tool_calls': [{'function': {'arguments': '{"answer": "They weigh the same.", "justification": "A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount."}', 'name': 'AnswerWithJustification'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'function'}]}, response_metadata={'finish_reason': 'tool_calls', 'usage': {'acceptance_rate': 5, 'completion_tokens': 53, 'completion_tokens_after_first_per_sec': 343.7964936837758, 'completion_tokens_after_first_per_sec_first_ten': 439.1205661878638, 'completion_tokens_per_sec': 162.8511306784833, 'end_time': 1731527851.0698032, 'is_last_response': True, 'prompt_tokens': 213, 'start_time': 1731527850.7137961, 'time_to_first_token': 0.20475482940673828, 'total_latency': 0.32545061111450196, 'total_tokens': 266, 'total_tokens_per_sec': 817.3283162354066}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731527850}, id='95667eaf-447f-4b53-bb6e-b6e1094ded88', tool_calls=[{'name': 'AnswerWithJustification', 'args': {'answer': 'They weigh the same.', 'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'}, 'id': 'call_17a431fc6a4240e1bd', 'type': 'tool_call'}]),
                #     'parsed': AnswerWithJustification(answer='They weigh the same.', justification='A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'),
                #     'parsing_error': None
                # }

        Example: schema=TypedDict class, method="function_calling", include_raw=False:
            .. code-block:: python

                # IMPORTANT: If you are using Python <=3.8, you need to import Annotated
                # from typing_extensions, not from typing.
                from typing_extensions import Annotated, TypedDict

                from langchain_community.chat_models import ChatSambaStudio


                class AnswerWithJustification(TypedDict):
                    '''An answer to the user question along with justification for the answer.'''

                    answer: str
                    justification: Annotated[
                        Optional[str], None, "A justification for the answer."
                    ]


                llm = ChatSambaStudio(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=OpenAI function schema, method="function_calling", include_raw=False:
            .. code-block:: python

                from langchain_community.chat_models import ChatSambaStudio

                oai_schema = {
                    'name': 'AnswerWithJustification',
                    'description': 'An answer to the user question along with justification for the answer.',
                    'parameters': {
                        'type': 'object',
                        'properties': {
                            'answer': {'type': 'string'},
                            'justification': {'description': 'A justification for the answer.', 'type': 'string'}
                        },
                       'required': ['answer']
                   }
                }

                llm = ChatSambaStudio(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(oai_schema)

                structured_llm.invoke(
                    "What weighs more a pound of bricks or a pound of feathers"
                )
                # -> {
                #     'answer': 'They weigh the same',
                #     'justification': 'A pound is a unit of weight or mass, so one pound of bricks and one pound of feathers both weigh the same amount.'
                # }

        Example: schema=Pydantic class, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_community.chat_models import ChatSambaStudio
                from pydantic import BaseModel

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaStudio(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(
                    AnswerWithJustification,
                    method="json_mode",
                    include_raw=True
                )

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_mode", include_raw=True:
            .. code-block::

                from langchain_community.chat_models import ChatSambaStudio

                llm = ChatSambaStudio(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(method="json_mode", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 4.722222222222222, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 357.1315485254867, 'completion_tokens_after_first_per_sec_first_ten': 416.83279609305305, 'completion_tokens_per_sec': 240.92819585198137, 'end_time': 1731528164.8474727, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528164.4906917, 'time_to_first_token': 0.13837409019470215, 'total_latency': 0.3278985247892492, 'total_tokens': 149, 'total_tokens_per_sec': 454.4088757208256}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528164}, id='15261eaf-8a25-42ef-8ed5-f63d8bf5b1b0'),
                #     'parsed': {
                #         'answer': 'They are the same weight',
                #         'justification': 'A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'},
                #     },
                #     'parsing_error': None
                # }

        Example: schema=None, method="json_schema", include_raw=True:
            .. code-block::

                from langchain_community.chat_models import ChatSambaStudio

                class AnswerWithJustification(BaseModel):
                    answer: str
                    justification: str

                llm = ChatSambaStudio(model="Meta-Llama-3.1-70B-Instruct", temperature=0)
                structured_llm = llm.with_structured_output(AnswerWithJustification, method="json_schema", include_raw=True)

                structured_llm.invoke(
                    "Answer the following question. "
                    "Make sure to return a JSON blob with keys 'answer' and 'justification'.\n\n"
                    "What's heavier a pound of bricks or a pound of feathers?"
                )
                # -> {
                #     'raw': AIMessage(content='{\n  "answer": "They are the same weight",\n  "justification": "A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities."\n}', additional_kwargs={}, response_metadata={'finish_reason': 'stop', 'usage': {'acceptance_rate': 5.3125, 'completion_tokens': 79, 'completion_tokens_after_first_per_sec': 292.65701089829776, 'completion_tokens_after_first_per_sec_first_ten': 346.43324678555325, 'completion_tokens_per_sec': 200.012158915008, 'end_time': 1731528071.1708555, 'is_last_response': True, 'prompt_tokens': 70, 'start_time': 1731528070.737394, 'time_to_first_token': 0.16693782806396484, 'total_latency': 0.3949759876026827, 'total_tokens': 149, 'total_tokens_per_sec': 377.2381225105847}, 'model_name': 'Meta-Llama-3.1-70B-Instruct', 'system_fingerprint': 'fastcoe', 'created': 1731528070}, id='83208297-3eb9-4021-a856-ca78a15758df'),
                #     'parsed': AnswerWithJustification(answer='They are the same weight', justification='A pound is a unit of weight or mass, so a pound of bricks and a pound of feathers both weigh the same amount, one pound. The difference is in their density and volume. A pound of feathers would take up more space than a pound of bricks due to the difference in their densities.'),
                #     'parsing_error': None
                # }

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
            tool_name = convert_to_openai_tool(schema)["function"]["name"]
            llm = self.bind_tools([schema], tool_choice=tool_name)
            if is_pydantic_schema:
                output_parser: OutputParserLike[Any] = PydanticToolsParser(
                    tools=[schema],  # type: ignore[list-item]
                    first_tool_only=True,
                )
            else:
                output_parser = JsonOutputKeyToolsParser(
                    key_name=tool_name, first_tool_only=True
                )
        elif method == "json_mode":
            llm = self
            # TODO bind response format when json mode available by API
            # llm = self.bind(response_format={"type": "json_object"})
            if is_pydantic_schema:
                schema = cast(Type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()

        elif method == "json_schema":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is not 'json_mode'. "
                    "Received None."
                )
            llm = self
            # TODO bind response format when json schema available by API,
            # update example
            # llm = self.bind(
            #   response_format={"type": "json_object", "json_schema": schema}
            # )
            if is_pydantic_schema:
                schema = cast(Type[BaseModel], schema)
                output_parser = PydanticOutputParser(pydantic_object=schema)
            else:
                output_parser = JsonOutputParser()
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected one of 'function_calling' or "
                f"'json_mode'. Received: '{method}'"
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

    def _get_role(self, message: BaseMessage) -> str:
        """
        Get the role of LangChain BaseMessage

        Args:
            message: LangChain BaseMessage

        Returns:
            str: Role of the LangChain BaseMessage
        """
        if isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        elif isinstance(message, ChatMessage):
            role = message.role
        else:
            raise TypeError(f"Got unknown type {message}")
        return role

    def _messages_to_string(self, messages: List[BaseMessage], **kwargs: Any) -> str:
        """
        Convert a list of BaseMessages to a:
        - dumped json string with Role / content dict structure
            when process_prompt is true,
        - string with special tokens if process_prompt is false
        for generic V1 and V2 endpoints

        Args:
            messages: list of BaseMessages

        Returns:
            str: string to send as model input depending on process_prompt param
        """
        if self.process_prompt:
            messages_dict: Dict[str, Any] = {
                "conversation_id": "sambaverse-conversation-id",
                "messages": [],
                **kwargs,
            }
            for message in messages:
                if isinstance(message, AIMessage):
                    message_dict = {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content,
                    }
                    if "tool_calls" in message.additional_kwargs:
                        message_dict["tool_calls"] = message.additional_kwargs[
                            "tool_calls"
                        ]
                        if message_dict["content"] == "":
                            message_dict["content"] = None

                elif isinstance(message, ToolMessage):
                    message_dict = {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content,
                        "tool_call_id": message.tool_call_id,
                    }

                else:
                    message_dict = {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content,
                    }

                messages_dict["messages"].append(message_dict)

            messages_string = json.dumps(messages_dict)

        else:
            if "tools" in kwargs.keys():
                raise NotImplementedError(
                    "tool calling not supported in API Generic V2 "
                    "without process_prompt, switch to OpenAI compatible API "
                    "or Generic V2 API with process_prompt=True"
                )
            messages_string = self.special_tokens["start"]
            for message in messages:
                messages_string += self.special_tokens["start_role"].format(
                    role=self._get_role(message)
                )
                messages_string += f" {message.content} "
                messages_string += self.special_tokens["end_role"]
            messages_string += self.special_tokens["end"]

        return messages_string

    def _get_sambastudio_urls(self, url: str) -> Tuple[str, str]:
        """
        Get streaming and non streaming URLs from the given URL

        Args:
            url: string with sambastudio base or streaming endpoint url

        Returns:
            base_url: string with url to do non streaming calls
            streaming_url: string with url to do streaming calls
        """
        if "chat/completions" in url:
            base_url = url
            stream_url = url
        else:
            if "stream" in url:
                base_url = url.replace("stream/", "")
                stream_url = url
            else:
                base_url = url
                if "generic" in url:
                    stream_url = "generic/stream".join(url.split("generic"))
                else:
                    raise ValueError("Unsupported URL")
        return base_url, stream_url

    def _handle_request(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        streaming: Optional[bool] = False,
        **kwargs: Any,
    ) -> Response:
        """
        Performs a post request to the LLM API.

        Args:
        messages_dicts: List of role / content dicts to use as input.
        stop: list of stop tokens
        streaming: wether to do a streaming call

        Returns:
            A request Response object
        """

        # create request payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            messages_dicts = _create_message_dicts(messages)
            data = {
                "messages": messages_dicts,
                "max_tokens": self.max_tokens,
                "stop": stop,
                "model": self.model,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "stream": streaming,
                "stream_options": self.stream_options,
                **kwargs,
            }
            data = {key: value for key, value in data.items() if value is not None}
            headers = {
                "Authorization": f"Bearer "
                f"{self.sambastudio_api_key.get_secret_value()}",
                "Content-Type": "application/json",
                **self.additional_headers,
            }

        # create request payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            items = [
                {"id": "item0", "value": self._messages_to_string(messages, **kwargs)}
            ]
            params: Dict[str, Any] = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
            }
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            params = {key: value for key, value in params.items() if value is not None}
            data = {"items": items, "params": params}
            headers = {
                "key": self.sambastudio_api_key.get_secret_value(),
                **self.additional_headers,
            }

        # create request payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            if "tools" in kwargs.keys():
                raise NotImplementedError(
                    "tool calling not supported in API Generic V1, "
                    "switch to OpenAI compatible API or Generic V2 API"
                )
            params = {
                "select_expert": self.model,
                "process_prompt": self.process_prompt,
                "max_tokens_to_generate": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "do_sample": self.do_sample,
                **kwargs,
            }
            if self.model_kwargs is not None:
                params = {**params, **self.model_kwargs}
            params = {
                key: {"type": type(value).__name__, "value": str(value)}
                for key, value in params.items()
                if value is not None
            }
            if streaming:
                data = {
                    "instance": self._messages_to_string(messages),
                    "params": params,
                }
            else:
                data = {
                    "instances": [self._messages_to_string(messages)],
                    "params": params,
                }
            headers = {
                "key": self.sambastudio_api_key.get_secret_value(),
                **self.additional_headers,
            }

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        http_session = requests.Session()
        if streaming:
            response = http_session.post(
                self.streaming_url, headers=headers, json=data, stream=True
            )
        else:
            response = http_session.post(
                self.base_url, headers=headers, json=data, stream=False
            )
        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}."
                f"{response.text}."
            )
        return response

    def _process_response(self, response: Response) -> AIMessage:
        """
        Process a non streaming response from the api

        Args:
            response: A request Response object

        Returns
            generation: an AIMessage with model generation
        """

        # Extract json payload form response
        try:
            response_dict = response.json()
        except Exception as e:
            raise RuntimeError(
                f"Sambanova /complete call failed couldn't get JSON response {e}"
                f"response: {response.text}"
            )

        additional_kwargs: Dict[str, Any] = {}
        tool_calls = []
        invalid_tool_calls = []

        # process response payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            content = response_dict["choices"][0]["message"].get("content", "")
            if content is None:
                content = ""
            id = response_dict["id"]
            response_metadata = {
                "finish_reason": response_dict["choices"][0]["finish_reason"],
                "usage": response_dict.get("usage"),
                "model_name": response_dict["model"],
                "system_fingerprint": response_dict["system_fingerprint"],
                "created": response_dict["created"],
            }
            raw_tool_calls = response_dict["choices"][0]["message"].get("tool_calls")
            if raw_tool_calls:
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    if isinstance(raw_tool_call["function"]["arguments"], dict):
                        raw_tool_call["function"]["arguments"] = json.dumps(
                            raw_tool_call["function"].get("arguments", {})
                        )
                    try:
                        tool_calls.append(
                            parse_tool_call(raw_tool_call, return_id=True)
                        )
                    except Exception as e:
                        invalid_tool_calls.append(
                            make_invalid_tool_call(raw_tool_call, str(e))
                        )

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            content = response_dict["items"][0]["value"]["completion"]
            id = response_dict["items"][0]["id"]
            response_metadata = response_dict["items"][0]
            raw_tool_calls = response_dict["items"][0]["value"].get("tool_calls")
            if raw_tool_calls:
                additional_kwargs["tool_calls"] = raw_tool_calls
                for raw_tool_call in raw_tool_calls:
                    if isinstance(raw_tool_call["function"]["arguments"], dict):
                        raw_tool_call["function"]["arguments"] = json.dumps(
                            raw_tool_call["function"].get("arguments", {})
                        )
                    try:
                        tool_calls.append(
                            parse_tool_call(raw_tool_call, return_id=True)
                        )
                    except Exception as e:
                        invalid_tool_calls.append(
                            make_invalid_tool_call(raw_tool_call, str(e))
                        )

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            content = response_dict["predictions"][0]["completion"]
            id = None
            response_metadata = response_dict

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

        return AIMessage(
            content=content,
            additional_kwargs=additional_kwargs,
            tool_calls=tool_calls,
            invalid_tool_calls=invalid_tool_calls,
            response_metadata=response_metadata,
            id=id,
        )

    def _process_stream_response(
        self, response: Response
    ) -> Iterator[BaseMessageChunk]:
        """
        Process a streaming response from the api

        Args:
            response: An iterable request Response object

        Yields:
            generation: an AIMessageChunk with model partial generation
        """

        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )

        # process response payload for openai compatible API
        if "chat/completions" in self.sambastudio_url:
            finish_reason = ""
            client = sseclient.SSEClient(response)
            for event in client.events():
                if event.event == "error_event":
                    raise RuntimeError(
                        f"Sambanova /complete call failed with status code "
                        f"{response.status_code}."
                        f"{event.data}."
                    )
                try:
                    # check if the response is not a final event ("[DONE]")
                    if event.data != "[DONE]":
                        if isinstance(event.data, str):
                            data = json.loads(event.data)
                        else:
                            raise RuntimeError(
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                        if data.get("error"):
                            raise RuntimeError(
                                f"Sambanova /complete call failed with status code "
                                f"{response.status_code}."
                                f"{event.data}."
                            )
                        if len(data["choices"]) > 0:
                            finish_reason = data["choices"][0].get("finish_reason")
                            content = data["choices"][0]["delta"]["content"]
                            id = data["id"]
                            metadata = {}
                        else:
                            content = ""
                            id = data["id"]
                            metadata = {
                                "finish_reason": finish_reason,
                                "usage": data.get("usage"),
                                "model_name": data["model"],
                                "system_fingerprint": data["system_fingerprint"],
                                "created": data["created"],
                            }
                        if data.get("usage") is not None:
                            content = ""
                            id = data["id"]
                            metadata = {
                                "finish_reason": finish_reason,
                                "usage": data.get("usage"),
                                "model_name": data["model"],
                                "system_fingerprint": data["system_fingerprint"],
                                "created": data["created"],
                            }
                        yield AIMessageChunk(
                            content=content,
                            id=id,
                            response_metadata=metadata,
                            additional_kwargs={},
                        )

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"data: {event.data}"
                    )

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            for line in response.iter_lines():
                try:
                    data = json.loads(line)
                    content = data["result"]["items"][0]["value"]["stream_token"]
                    id = data["result"]["items"][0]["id"]
                    if data["result"]["items"][0]["value"]["is_last_response"]:
                        metadata = {
                            "finish_reason": data["result"]["items"][0]["value"].get(
                                "stop_reason"
                            ),
                            "prompt": data["result"]["items"][0]["value"].get("prompt"),
                            "usage": {
                                "prompt_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("prompt_tokens_count"),
                                "completion_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("completion_tokens_count"),
                                "total_tokens_count": data["result"]["items"][0][
                                    "value"
                                ].get("total_tokens_count"),
                                "start_time": data["result"]["items"][0]["value"].get(
                                    "start_time"
                                ),
                                "end_time": data["result"]["items"][0]["value"].get(
                                    "end_time"
                                ),
                                "model_execution_time": data["result"]["items"][0][
                                    "value"
                                ].get("model_execution_time"),
                                "time_to_first_token": data["result"]["items"][0][
                                    "value"
                                ].get("time_to_first_token"),
                                "throughput_after_first_token": data["result"]["items"][
                                    0
                                ]["value"].get("throughput_after_first_token"),
                                "batch_size_used": data["result"]["items"][0][
                                    "value"
                                ].get("batch_size_used"),
                            },
                        }
                    else:
                        metadata = {}
                    yield AIMessageChunk(
                        content=content,
                        id=id,
                        response_metadata=metadata,
                        additional_kwargs={},
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )

        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            for line in response.iter_lines():
                try:
                    data = json.loads(line)
                    content = data["result"]["responses"][0]["stream_token"]
                    id = None
                    if data["result"]["responses"][0]["is_last_response"]:
                        metadata = {
                            "finish_reason": data["result"]["responses"][0].get(
                                "stop_reason"
                            ),
                            "prompt": data["result"]["responses"][0].get("prompt"),
                            "usage": {
                                "prompt_tokens_count": data["result"]["responses"][
                                    0
                                ].get("prompt_tokens_count"),
                                "completion_tokens_count": data["result"]["responses"][
                                    0
                                ].get("completion_tokens_count"),
                                "total_tokens_count": data["result"]["responses"][
                                    0
                                ].get("total_tokens_count"),
                                "start_time": data["result"]["responses"][0].get(
                                    "start_time"
                                ),
                                "end_time": data["result"]["responses"][0].get(
                                    "end_time"
                                ),
                                "model_execution_time": data["result"]["responses"][
                                    0
                                ].get("model_execution_time"),
                                "time_to_first_token": data["result"]["responses"][
                                    0
                                ].get("time_to_first_token"),
                                "throughput_after_first_token": data["result"][
                                    "responses"
                                ][0].get("throughput_after_first_token"),
                                "batch_size_used": data["result"]["responses"][0].get(
                                    "batch_size_used"
                                ),
                            },
                        }
                    else:
                        metadata = {}
                    yield AIMessageChunk(
                        content=content,
                        id=id,
                        response_metadata=metadata,
                        additional_kwargs={},
                    )

                except Exception as e:
                    raise RuntimeError(
                        f"Error getting content chunk raw streamed response: {e}"
                        f"line: {line}"
                    )

        else:
            raise ValueError(
                f"Unsupported URL{self.sambastudio_url}"
                "only openai, generic v1 and generic v2 APIs are supported"
            )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Call SambaStudio models.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.

        Returns:
            result: ChatResult with model generation
        """
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            if stream_iter:
                return generate_from_stream(stream_iter)
        response = self._handle_request(messages, stop, streaming=False, **kwargs)
        message = self._process_response(response)
        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stream the output of the SambaStudio model.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.

        Yields:
            chunk: ChatGenerationChunk with model partial generation
        """
        response = self._handle_request(messages, stop, streaming=True, **kwargs)
        for ai_message_chunk in self._process_stream_response(response):
            chunk = ChatGenerationChunk(message=ai_message_chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
