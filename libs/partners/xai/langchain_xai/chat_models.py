"""Wrapper around xAI's Chat Completions API."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, Optional, TypeVar, Union

import openai
from langchain_core.messages import AIMessageChunk
from langchain_core.utils import secret_from_env
from langchain_openai.chat_models.base import BaseChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import (
        LangSmithParams,
        LanguageModelInput,
    )
    from langchain_core.outputs import ChatGenerationChunk, ChatResult
    from langchain_core.runnables import Runnable

_BM = TypeVar("_BM", bound=BaseModel)
_DictOrPydanticClass = Union[dict[str, Any], type[_BM], type]
_DictOrPydantic = Union[dict, _BM]


class ChatXAI(BaseChatOpenAI):  # type: ignore[override]
    r"""ChatXAI chat model.

    Refer to `xAI's documentation <https://docs.x.ai/docs/api-reference#chat-completions>`__
    for more nuanced details on the API's behavior and supported parameters.

    Setup:
        Install ``langchain-xai`` and set environment variable ``XAI_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-xai
            export XAI_API_KEY="your-api-key"


    Key init args — completion params:
        model: str
            Name of model to use.
        temperature: float
            Sampling temperature between ``0`` and ``2``. Higher values mean more random completions,
            while lower values (like ``0.2``) mean more focused and deterministic completions.
            (Default: ``1``.)
        max_tokens: Optional[int]
            Max number of tokens to generate. Refer to your `model's documentation <https://docs.x.ai/docs/models#model-pricing>`__
            for the maximum number of tokens it can generate.
        logprobs: Optional[bool]
            Whether to return logprobs.

    Key init args — client params:
        timeout: Union[float, Tuple[float, float], Any, None]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            xAI API key. If not passed in will be read from env var ``XAI_API_KEY``.

    Instantiate:
        .. code-block:: python

            from langchain_xai import ChatXAI

            llm = ChatXAI(
                model="grok-4",
                temperature=0,
                max_tokens=None,
                timeout=None,
                max_retries=2,
                # api_key="...",
                # other params...
            )

    Invoke:
        .. code-block:: python

            messages = [
                (
                    "system",
                    "You are a helpful translator. Translate the user sentence to French.",
                ),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={
                    'token_usage': {'completion_tokens': 9, 'prompt_tokens': 32, 'total_tokens': 41},
                    'model_name': 'grok-4',
                    'system_fingerprint': None,
                    'finish_reason': 'stop',
                    'logprobs': None
                },
                id='run-168dceca-3b8b-4283-94e3-4c739dbc1525-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 9, 'total_tokens': 41})

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.text, end="")

        .. code-block:: python

            content='J' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content="'" id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ad' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ore' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content=' la' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content=' programm' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='ation' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='.' id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'
            content='' response_metadata={'finish_reason': 'stop', 'model_name': 'grok-4'} id='run-1bc996b5-293f-4114-96a1-e0f755c05eb9'


    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            AIMessage(
                content="J'adore la programmation.",
                response_metadata={
                    'token_usage': {'completion_tokens': 9, 'prompt_tokens': 32, 'total_tokens': 41},
                    'model_name': 'grok-4',
                    'system_fingerprint': None,
                    'finish_reason': 'stop',
                    'logprobs': None
                },
                id='run-09371a11-7f72-4c53-8e7c-9de5c238b34c-0',
                usage_metadata={'input_tokens': 32, 'output_tokens': 9, 'total_tokens': 41})

    Reasoning:
        `Certain xAI models <https://docs.x.ai/docs/models#model-pricing>`__ support reasoning,
        which allows the model to provide reasoning content along with the response.

        If provided, reasoning content is returned under the ``additional_kwargs`` field of the
        AIMessage or AIMessageChunk.

        If supported, reasoning effort can be specified in the model constructor's ``extra_body``
        argument, which will control the amount of reasoning the model does. The value can be one of
        ``'low'`` or ``'high'``.

        .. code-block:: python

            model = ChatXAI(
                model="grok-3-mini",
                extra_body={"reasoning_effort": "high"},
            )

        .. note::
            As of 2025-07-10, ``reasoning_content`` is only returned in Grok 3 models, such as
            `Grok 3 Mini <https://docs.x.ai/docs/models/grok-3-mini>`__.

        .. note::
            Note that in `Grok 4 <https://docs.x.ai/docs/models/grok-4-0709>`__, as of 2025-07-10,
            reasoning is not exposed in ``reasoning_content`` (other than initial ``'Thinking...'`` text),
            reasoning cannot be disabled, and the ``reasoning_effort`` cannot be specified.

    Tool calling / function calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            llm = ChatXAI(model="grok-4")

            class GetWeather(BaseModel):
                '''Get the current weather in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )

            class GetPopulation(BaseModel):
                '''Get the current population in a given location'''

                location: str = Field(
                    ..., description="The city and state, e.g. San Francisco, CA"
                )

            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke(
                "Which city is bigger: LA or NY?"
            )
            ai_msg.tool_calls

        .. code-block:: python

            [
                {
                    'name': 'GetPopulation',
                    'args': {'location': 'NY'},
                    'id': 'call_m5tstyn2004pre9bfuxvom8x',
                    'type': 'tool_call'
                },
                {
                    'name': 'GetPopulation',
                    'args': {'location': 'LA'},
                    'id': 'call_0vjgq455gq1av5sp9eb1pw6a',
                    'type': 'tool_call'
                }
            ]

        .. note::
            With stream response, the tool / function call will be returned in whole in a
            single chunk, instead of being streamed across chunks.

        Tool choice can be controlled by setting the ``tool_choice`` parameter in the model
        constructor's ``extra_body`` argument. For example, to disable tool / function calling:
        .. code-block:: python

            llm = ChatXAI(model="grok-4", extra_body={"tool_choice": "none"})

        To require that the model always calls a tool / function, set ``tool_choice`` to ``'required'``:

        .. code-block:: python

            llm = ChatXAI(model="grok-4", extra_body={"tool_choice": "required"})

        To specify a tool / function to call, set ``tool_choice`` to the name of the tool / function:

        .. code-block:: python

            from pydantic import BaseModel, Field

            llm = ChatXAI(
                model="grok-4",
                extra_body={
                    "tool_choice": {"type": "function", "function": {"name": "GetWeather"}}
                },
            )

            class GetWeather(BaseModel):
                \"\"\"Get the current weather in a given location\"\"\"

                location: str = Field(..., description='The city and state, e.g. San Francisco, CA')


            class GetPopulation(BaseModel):
                \"\"\"Get the current population in a given location\"\"\"

                location: str = Field(..., description='The city and state, e.g. San Francisco, CA')


            llm_with_tools = llm.bind_tools([GetWeather, GetPopulation])
            ai_msg = llm_with_tools.invoke(
                "Which city is bigger: LA or NY?",
            )
            ai_msg.tool_calls

        The resulting tool call would be:

        .. code-block:: python

            [{'name': 'GetWeather',
            'args': {'location': 'Los Angeles, CA'},
            'id': 'call_81668711',
            'type': 'tool_call'}]

    Parallel tool calling / parallel function calling:
        By default, parallel tool / function calling is enabled, so you can process
        multiple function calls in one request/response cycle. When two or more tool calls
        are required, all of the tool call requests will be included in the response body.

    Structured output:
        .. code-block:: python

            from typing import Optional

            from pydantic import BaseModel, Field


            class Joke(BaseModel):
                '''Joke to tell user.'''

                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")
                rating: Optional[int] = Field(description="How funny the joke is, from 1 to 10")


            structured_llm = llm.with_structured_output(Joke)
            structured_llm.invoke("Tell me a joke about cats")

        .. code-block:: python

            Joke(
                setup='Why was the cat sitting on the computer?',
                punchline='To keep an eye on the mouse!',
                rating=7
            )

    Live Search:
        xAI supports a `Live Search <https://docs.x.ai/docs/guides/live-search>`__
        feature that enables Grok to ground its answers using results from web searches.

        .. code-block:: python

            from langchain_xai import ChatXAI

            llm = ChatXAI(
                model="grok-4",
                search_parameters={
                    "mode": "auto",
                    # Example optional parameters below:
                    "max_search_results": 3,
                    "from_date": "2025-05-26",
                    "to_date": "2025-05-27",
                }
            )

            llm.invoke("Provide me a digest of world news in the last 24 hours.")

        .. note::
            `Citations <https://docs.x.ai/docs/guides/live-search#returning-citations>`__
            are only available in `Grok 3 <https://docs.x.ai/docs/models/grok-3>`__.

    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 37, 'output_tokens': 6, 'total_tokens': 43}

    Logprobs:
        .. code-block:: python

            logprobs_llm = llm.bind(logprobs=True)
            messages=[("human","Say Hello World! Do not return anything else.")]
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

            {
                'content': None,
                'token_ids': [22557, 3304, 28808, 2],
                'tokens': [' Hello', ' World', '!', '</s>'],
                'token_logprobs': [-4.7683716e-06, -5.9604645e-07, 0, -0.057373047]
            }

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

            {
                'token_usage': {
                    'completion_tokens': 4,
                    'prompt_tokens': 19,
                    'total_tokens': 23
                    },
                'model_name': 'grok-4',
                'system_fingerprint': None,
                'finish_reason': 'stop',
                'logprobs': None
            }

    """  # noqa: E501

    model_name: str = Field(default="grok-4", alias="model")
    """Model name to use."""
    xai_api_key: Optional[SecretStr] = Field(
        alias="api_key",
        default_factory=secret_from_env("XAI_API_KEY", default=None),
    )
    """xAI API key.

    Automatically read from env variable ``XAI_API_KEY`` if not provided.
    """
    xai_api_base: str = Field(default="https://api.x.ai/v1/")
    """Base URL path for API requests."""
    search_parameters: Optional[dict[str, Any]] = None
    """Parameters for search requests. Example: ``{"mode": "auto"}``."""

    openai_api_key: Optional[SecretStr] = None
    openai_api_base: Optional[str] = None

    model_config = ConfigDict(
        populate_by_name=True,
    )

    @property
    def lc_secrets(self) -> dict[str, str]:
        """A map of constructor argument names to secret ids.

        For example, ``{"xai_api_key": "XAI_API_KEY"}``
        """
        return {"xai_api_key": "XAI_API_KEY"}

    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object."""
        return ["langchain_xai", "chat_models"]

    @property
    def lc_attributes(self) -> dict[str, Any]:
        """List of attribute names that should be included in the serialized kwargs.

        These attributes must be accepted by the constructor.
        """
        attributes: dict[str, Any] = {}

        if self.xai_api_base:
            attributes["xai_api_base"] = self.xai_api_base

        return attributes

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by LangChain."""
        return True

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "xai-chat"

    def _get_ls_params(
        self, stop: Optional[list[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "xai"
        return params

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            msg = "n must be at least 1."
            raise ValueError(msg)
        if self.n is not None and self.n > 1 and self.streaming:
            msg = "n must be 1 when streaming."
            raise ValueError(msg)

        client_params: dict = {
            "api_key": (
                self.xai_api_key.get_secret_value() if self.xai_api_key else None
            ),
            "base_url": self.xai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if client_params["api_key"] is None:
            msg = (
                "xAI API key is not set. Please set it in the `xai_api_key` field or "
                "in the `XAI_API_KEY` environment variable."
            )
            raise ValueError(msg)

        if not (self.client or None):
            sync_specific: dict = {"http_client": self.http_client}
            self.client = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
            self.root_client = openai.OpenAI(**client_params, **sync_specific)
        if not (self.async_client or None):
            async_specific: dict = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,
                **async_specific,
            )
        return self

    @property
    def _default_params(self) -> dict[str, Any]:
        """Get default parameters."""
        params = super()._default_params
        if self.search_parameters:
            if "extra_body" in params:
                params["extra_body"]["search_parameters"] = self.search_parameters
            else:
                params["extra_body"] = {"search_parameters": self.search_parameters}

        return params

    def _create_chat_result(
        self,
        response: Union[dict, openai.BaseModel],
        generation_info: Optional[dict] = None,
    ) -> ChatResult:
        rtn = super()._create_chat_result(response, generation_info)

        if not isinstance(response, openai.BaseModel):
            return rtn

        if hasattr(response.choices[0].message, "reasoning_content"):  # type: ignore[attr-defined]
            rtn.generations[0].message.additional_kwargs["reasoning_content"] = (
                response.choices[0].message.reasoning_content  # type: ignore[attr-defined]
            )

        if hasattr(response, "citations"):
            rtn.generations[0].message.additional_kwargs["citations"] = (
                response.citations
            )

        return rtn

    def _convert_chunk_to_generation_chunk(
        self,
        chunk: dict,
        default_chunk_class: type,
        base_generation_info: Optional[dict],
    ) -> Optional[ChatGenerationChunk]:
        generation_chunk = super()._convert_chunk_to_generation_chunk(
            chunk,
            default_chunk_class,
            base_generation_info,
        )
        if (choices := chunk.get("choices")) and generation_chunk:
            top = choices[0]
            if isinstance(generation_chunk.message, AIMessageChunk) and (
                reasoning_content := top.get("delta", {}).get("reasoning_content")
            ):
                generation_chunk.message.additional_kwargs["reasoning_content"] = (
                    reasoning_content
                )

        if (
            (citations := chunk.get("citations"))
            and generation_chunk
            and isinstance(generation_chunk.message, AIMessageChunk)
        ):
            generation_chunk.message.additional_kwargs["citations"] = citations

        return generation_chunk

    def with_structured_output(
        self,
        schema: Optional[_DictOrPydanticClass] = None,
        *,
        method: Literal[
            "function_calling", "json_mode", "json_schema"
        ] = "function_calling",
        include_raw: bool = False,
        strict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, _DictOrPydantic]:
        """Model wrapper that returns outputs formatted to match the given schema.

        Args:
            schema: The output schema. Can be passed in as:

                - an OpenAI function/tool schema,
                - a JSON Schema,
                - a TypedDict class (support added in 0.1.20),
                - or a Pydantic class.

                If ``schema`` is a Pydantic class then the model output will be a
                Pydantic instance of that class, and the model-generated fields will be
                validated by the Pydantic class. Otherwise the model output will be a
                dict and will not be validated. See :meth:`langchain_core.utils.function_calling.convert_to_openai_tool`
                for more on how to properly specify types and descriptions of
                schema fields when specifying a Pydantic or TypedDict class.

            method: The method for steering model generation, one of:

                - ``'function_calling'``:
                    Uses xAI's `tool-calling features <https://docs.x.ai/docs/guides/function-calling>`__.
                - ``'json_schema'``:
                    Uses xAI's `structured output feature <https://docs.x.ai/docs/guides/structured-outputs>`__.
                - ``'json_mode'``:
                    Uses xAI's JSON mode feature.

            include_raw:
                If ``False`` then only the parsed structured output is returned. If
                an error occurs during model output parsing it will be raised. If ``True``
                then both the raw model response (a BaseMessage) and the parsed model
                response will be returned. If an error occurs during output parsing it
                will be caught and returned as well. The final output is always a dict
                with keys ``'raw'``, ``'parsed'``, and ``'parsing_error'``.

            strict:
                - ``True``:
                    Model output is guaranteed to exactly match the schema.
                    The input schema will also be validated according to the `supported schemas <https://platform.openai.com/docs/guides/structured-outputs/supported-schemas?api-mode=responses#supported-schemas>`__.
                - ``False``:
                    Input schema will not be validated and model output will not be
                    validated.
                - ``None``:
                    ``strict`` argument will not be passed to the model.

            kwargs: Additional keyword args aren't supported.

        Returns:
            A Runnable that takes same inputs as a :class:`langchain_core.language_models.chat.BaseChatModel`.

            If ``include_raw`` is ``False`` and ``schema`` is a Pydantic class, Runnable outputs an instance of ``schema`` (i.e., a Pydantic object). Otherwise, if ``include_raw`` is ``False`` then Runnable outputs a dict.

            If ``include_raw`` is ``True``, then Runnable outputs a dict with keys:

            - ``'raw'``: BaseMessage
            - ``'parsed'``: None if there was a parsing error, otherwise the type depends on the ``schema`` as described above.
            - ``'parsing_error'``: Optional[BaseException]

        """  # noqa: E501
        # Some applications require that incompatible parameters (e.g., unsupported
        # methods) be handled.
        if method == "function_calling" and strict:
            strict = None
        return super().with_structured_output(
            schema, method=method, include_raw=include_raw, strict=strict, **kwargs
        )
