import json
import logging
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import (
    BaseChatModel,
    agenerate_from_stream,
    generate_from_stream,
)
from langchain_core.messages import AIMessageChunk, BaseMessage, BaseMessageChunk
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_community.chat_models.openai import _convert_delta_to_message_chunk
from langchain_community.llms.oci_data_science_model_deployment_endpoint import (
    DEFAULT_MODEL_NAME,
    BaseOCIModelDeployment,
)

logger = logging.getLogger(__name__)


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


class ChatOCIModelDeployment(BaseChatModel, BaseOCIModelDeployment):
    """OCI Data Science Model Deployment chat model integration.

    To use, you must provide the model HTTP endpoint from your deployed
    chat model, e.g. https://modeldeployment.<region>.oci.customer-oci.com/<md_ocid>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatOCIModelDeployment

            chat = ChatOCIModelDeployment(
                endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
                model="odsc-llm",
                streaming=True,
                max_retries=3,
                model_kwargs={
                    "max_token": 512,
                    "temperature": 0.2,
                    # other model parameters ...
                },
            )

    Invocation:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "Hello World!"),
            ]
            chat.invoke(messages)

        .. code-block:: python

            AIMessage(
                content='Bonjour le monde!',response_metadata={'token_usage': {'prompt_tokens': 40, 'total_tokens': 50, 'completion_tokens': 10},'model_name': 'odsc-llm','system_fingerprint': '','finish_reason': 'stop'},id='run-cbed62da-e1b3-4abd-9df3-ec89d69ca012-0')

    Streaming:
        .. code-block:: python

            for chunk in chat.stream(messages):
                print(chunk)

        .. code-block:: python

            content='' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content='\n' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content='B' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content='on' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content='j' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content='our' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content=' le' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content=' monde' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content='!' id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'
            content='' response_metadata={'finish_reason': 'stop'} id='run-23df02c6-c43f-42de-87c6-8ad382e125c3'

    Asyc:
        .. code-block:: python

            await chat.ainvoke(messages)

            # stream:
            # async for chunk in (await chat.astream(messages))

        .. code-block:: python

            AIMessage(content='Bonjour le monde!', response_metadata={'finish_reason': 'stop'}, id='run-8657a105-96b7-4bb6-b98e-b69ca420e5d1-0')

    Structured output:
        .. code-block:: python

            from typing import Optional

            from langchain_core.pydantic_v1 import BaseModel, Field

            class Joke(BaseModel):
                setup: str = Field(description="The setup of the joke")
                punchline: str = Field(description="The punchline to the joke")

            structured_llm = chat.with_structured_output(Joke, method="json_mode")
            structured_llm.invoke(
                "Tell me a joke about cats, respond in JSON with `setup` and `punchline` keys"
            )

        .. code-block:: python

            Joke(setup='Why did the cat get stuck in the tree?',punchline='Because it was chasing its tail!')

        See ``ChatOCIModelDeployment.with_structured_output()`` for more.

    Customized Usage:

    You can inherit from base class and overrwrite the `_process_response`, `_process_stream_response`,
    `_construct_json_body` for satisfying customized needed.

        .. code-block:: python

            class MyChatModel(ChatOCIModelDeployment):
                def _process_stream_response(self, response_json: dict) -> ChatGenerationChunk:
                    print("My customized streaming result handler.")
                    return GenerationChunk(...)

                def _process_response(self, response_json:dict) -> ChatResult:
                    print("My customized output handler.")
                    return ChatResult(...)

                def _construct_json_body(self, messages: list, params: dict) -> dict:
                    print("My customized payload handler.")
                    return {
                        "messages": messages,
                        **params,
                    }

            chat = MyChatModel(
                endpoint=f"https://modeldeployment.us-ashburn-1.oci.customer-oci.com/{ocid}/predict",
                model="odsc-llm",
            }

            chat.invoke("tell me a joke")

    """  # noqa: E501

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""

    model: str = DEFAULT_MODEL_NAME
    """The name of the model."""

    stop: Optional[List[str]] = None
    """Stop words to use when generating. Model output is cut off
    at the first occurrence of any of these substrings."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_model_depolyment_chat_endpoint"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        _model_kwargs = self.model_kwargs or {}
        return {
            **{"endpoint": self.endpoint, "model_kwargs": _model_kwargs},
            **self._default_params,
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return {
            "model": self.model,
            "stop": self.stop,
            "stream": self.streaming,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to an OCI Model Deployment Online endpoint.

        Args:
            messages:  The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.

        Returns:
            LangChain ChatResult

        Raises:
            RuntimeError:
                Raise when invoking endpoint fails.

        Example:

            .. code-block:: python

                messages = [
                    (
                        "system",
                        "You are a helpful assistant that translates English to French. Translate the user sentence.",
                    ),
                    ("human", "Hello World!"),
                ]

                response = chat.invoke(messages)
        """  # noqa: E501
        if self.streaming:
            stream_iter = self._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return generate_from_stream(stream_iter)

        requests_kwargs = kwargs.pop("requests_kwargs", {})
        params = self._invocation_params(stop, **kwargs)
        body = self._construct_json_body(messages, params)
        res = self.completion_with_retry(
            data=body, run_manager=run_manager, **requests_kwargs
        )
        return self._process_response(res.json())

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """Stream OCI Data Science Model Deployment endpoint on given messages.

        Args:
            messages (List[BaseMessage]):
                The messagaes to pass into the model.
            stop (List[str], Optional):
                List of stop words to use when generating.
            kwargs:
                requests_kwargs:
                    Additional ``**kwargs`` to pass to requests.post

        Returns:
            An iterator of ChatGenerationChunk.

        Raises:
            RuntimeError:
                Raise when invoking endpoint fails.

        Example:

            .. code-block:: python

                messages = [
                    (
                        "system",
                        "You are a helpful assistant that translates English to French. Translate the user sentence.",
                    ),
                    ("human", "Hello World!"),
                ]

                chunk_iter = chat.stream(messages)

        """  # noqa: E501
        requests_kwargs = kwargs.pop("requests_kwargs", {})
        self.streaming = True
        params = self._invocation_params(stop, **kwargs)
        body = self._construct_json_body(messages, params)  # request json body

        response = self.completion_with_retry(
            data=body, run_manager=run_manager, stream=True, **requests_kwargs
        )
        default_chunk_class = AIMessageChunk
        for line in self._parse_stream(response.iter_lines()):
            chunk = self._handle_sse_line(line, default_chunk_class)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Asynchronously call out to OCI Data Science Model Deployment
        endpoint on given messages.

        Args:
            messages (List[BaseMessage]):
                The messagaes to pass into the model.
            stop (List[str], Optional):
                List of stop words to use when generating.
            kwargs:
                requests_kwargs:
                    Additional ``**kwargs`` to pass to requests.post

        Returns:
            LangChain ChatResult.

        Raises:
            ValueError:
                Raise when invoking endpoint fails.

        Example:

            .. code-block:: python

                messages = [
                    (
                        "system",
                        "You are a helpful assistant that translates English to French. Translate the user sentence.",
                    ),
                    ("human", "I love programming."),
                ]

                resp = await chat.ainvoke(messages)

        """  # noqa: E501
        if self.streaming:
            stream_iter = self._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
            return await agenerate_from_stream(stream_iter)

        requests_kwargs = kwargs.pop("requests_kwargs", {})
        params = self._invocation_params(stop, **kwargs)
        body = self._construct_json_body(messages, params)
        response = await self.acompletion_with_retry(
            data=body,
            run_manager=run_manager,
            **requests_kwargs,
        )
        return self._process_response(response)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Asynchronously streaming OCI Data Science Model Deployment
        endpoint on given messages.

        Args:
            messages (List[BaseMessage]):
                The messagaes to pass into the model.
            stop (List[str], Optional):
                List of stop words to use when generating.
            kwargs:
                requests_kwargs:
                    Additional ``**kwargs`` to pass to requests.post

        Returns:
            An Asynciterator of ChatGenerationChunk.

        Raises:
            ValueError:
                Raise when invoking endpoint fails.

        Example:

            .. code-block:: python

                messages = [
                    (
                        "system",
                        "You are a helpful assistant that translates English to French. Translate the user sentence.",
                    ),
                    ("human", "I love programming."),
                ]

                chunk_iter = await chat.astream(messages)

        """  # noqa: E501
        requests_kwargs = kwargs.pop("requests_kwargs", {})
        self.streaming = True
        params = self._invocation_params(stop, **kwargs)
        body = self._construct_json_body(messages, params)  # request json body

        default_chunk_class = AIMessageChunk
        async for line in await self.acompletion_with_retry(
            data=body, run_manager=run_manager, stream=True, **requests_kwargs
        ):
            chunk = self._handle_sse_line(line, default_chunk_class)
            if run_manager:
                await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk

    def with_structured_output(
        self,
        schema: Optional[Union[Dict, Type[BaseModel]]] = None,
        *,
        method: Literal["json_mode"] = "json_mode",
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
            method: The method for steering model generation, currently only support
                for "json_mode". If "json_mode" then JSON mode will be used. Note that
                if using "json_mode" then you must include instructions for formatting
                the output into the desired schema into the model call.
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

        """  # noqa: E501
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "json_mode":
            llm = self.bind(response_format={"type": "json_object"})
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)  # type: ignore[type-var, arg-type]
                if is_pydantic_schema
                else JsonOutputParser()
            )
        else:
            raise ValueError(
                f"Unrecognized method argument. Expected `json_mode`."
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

    def _invocation_params(self, stop: Optional[List[str]], **kwargs: Any) -> dict:
        """Combines the invocation parameters with default parameters."""
        params = self._default_params
        _model_kwargs = self.model_kwargs or {}
        params["stop"] = stop or params.get("stop", [])
        return {**params, **_model_kwargs, **kwargs}

    def _handle_sse_line(
        self, line: str, default_chunk_cls: Type[BaseMessageChunk] = AIMessageChunk
    ) -> ChatGenerationChunk:
        """Handle a single Server-Sent Events (SSE) line and process it into
        a chat generation chunk.

        Args:
            line (str): A single line from the SSE stream in string format.
            default_chunk_cls (AIMessageChunk): The default class for message
                chunks to be used during the processing of the stream response.

        Returns:
            ChatGenerationChunk: The processed chat generation chunk. If an error
                occurs, an empty `ChatGenerationChunk` is returned.
        """
        try:
            obj = json.loads(line)
            return self._process_stream_response(obj, default_chunk_cls)
        except Exception as e:
            logger.debug(f"Error occurs when processing line={line}: {str(e)}")
            return ChatGenerationChunk(message=AIMessageChunk(content=""))

    def _construct_json_body(self, messages: list, params: dict) -> dict:
        """Constructs the request body as a dictionary (JSON).

        Args:
            messages (list): A list of message objects to be included in the
                request body.
            params (dict): A dictionary of additional parameters to be included
                in the request body.

        Returns:
            dict: A dictionary representing the JSON request body, including
                converted messages and additional parameters.

        """
        return {
            "messages": [convert_message_to_dict(m) for m in messages],
            **params,
        }

    def _process_stream_response(
        self,
        response_json: dict,
        default_chunk_cls: Type[BaseMessageChunk] = AIMessageChunk,
    ) -> ChatGenerationChunk:
        """Formats streaming response in OpenAI spec.

        Args:
            response_json (dict): The JSON response from the streaming endpoint.
            default_chunk_cls (type, optional): The default class to use for
                creating message chunks. Defaults to `AIMessageChunk`.

        Returns:
            ChatGenerationChunk: An object containing the processed message
                chunk and any relevant generation information such as finish
                reason and usage.

        Raises:
            ValueError: If the response JSON is not well-formed or does not
                contain the expected structure.
        """
        try:
            choice = response_json["choices"][0]
            if not isinstance(choice, dict):
                raise TypeError("Endpoint response is not well formed.")
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(
                "Error while formatting response payload for chat model of type"
            ) from e

        chunk = _convert_delta_to_message_chunk(choice["delta"], default_chunk_cls)
        default_chunk_cls = chunk.__class__
        finish_reason = choice.get("finish_reason")
        usage = choice.get("usage")
        gen_info = {}
        if finish_reason is not None:
            gen_info.update({"finish_reason": finish_reason})
        if usage is not None:
            gen_info.update({"usage": usage})

        return ChatGenerationChunk(
            message=chunk, generation_info=gen_info if gen_info else None
        )

    def _process_response(self, response_json: dict) -> ChatResult:
        """Formats response in OpenAI spec.

        Args:
            response_json (dict): The JSON response from the chat model endpoint.

        Returns:
            ChatResult: An object containing the list of `ChatGeneration` objects
            and additional LLM output information.

        Raises:
            ValueError: If the response JSON is not well-formed or does not
            contain the expected structure.

        """
        generations = []
        try:
            choices = response_json["choices"]
            if not isinstance(choices, list):
                raise TypeError("Endpoint response is not well formed.")
        except (KeyError, TypeError) as e:
            raise ValueError(
                "Error while formatting response payload for chat model of type"
            ) from e

        for choice in choices:
            message = convert_dict_to_message(choice["message"])
            generation_info = dict(finish_reason=choice.get("finish_reason"))
            if "logprobs" in choice:
                generation_info["logprobs"] = choice["logprobs"]

            gen = ChatGeneration(
                message=message,
                generation_info=generation_info,
            )
            generations.append(gen)

        token_usage = response_json.get("usage", {})
        llm_output = {
            "token_usage": token_usage,
            "model_name": self.model,
            "system_fingerprint": response_json.get("system_fingerprint", ""),
        }
        return ChatResult(generations=generations, llm_output=llm_output)


class ChatOCIModelDeploymentVLLM(ChatOCIModelDeployment):
    """OCI large language chat models deployed with vLLM.

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:

        .. code-block:: python

            from langchain_community.chat_models import ChatOCIModelDeploymentVLLM

            chat = ChatOCIModelDeploymentVLLM(
                endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
                frequency_penalty=0.1,
                max_tokens=512,
                temperature=0.2,
                top_p=1.0,
                # other model parameters...
            )

    """  # noqa: E501

    frequency_penalty: float = 0.0
    """Penalizes repeated tokens according to frequency. Between 0 and 1."""

    logit_bias: Optional[Dict[str, float]] = None
    """Adjust the probability of specific tokens being generated."""

    max_tokens: Optional[int] = 256
    """The maximum number of tokens to generate in the completion."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    presence_penalty: float = 0.0
    """Penalizes repeated tokens. Between 0 and 1."""

    temperature: float = 0.2
    """What sampling temperature to use."""

    top_p: float = 1.0
    """Total probability mass of tokens to consider at each step."""

    best_of: Optional[int] = None
    """Generates best_of completions server-side and returns the "best"
    (the one with the highest log probability per token).
    """

    use_beam_search: Optional[bool] = False
    """Whether to use beam search instead of sampling."""

    top_k: Optional[int] = -1
    """Number of most likely tokens to consider at each step."""

    min_p: Optional[float] = 0.0
    """Float that represents the minimum probability for a token to be considered. 
    Must be in [0,1]. 0 to disable this."""

    repetition_penalty: Optional[float] = 1.0
    """Float that penalizes new tokens based on their frequency in the
    generated text. Values > 1 encourage the model to use new tokens."""

    length_penalty: Optional[float] = 1.0
    """Float that penalizes sequences based on their length. Used only
    when `use_beam_search` is True."""

    early_stopping: Optional[bool] = False
    """Controls the stopping condition for beam search. It accepts the
    following values: `True`, where the generation stops as soon as there
    are `best_of` complete candidates; `False`, where a heuristic is applied
    to the generation stops when it is very unlikely to find better candidates;
    `never`, where the beam search procedure only stops where there cannot be
    better candidates (canonical beam search algorithm)."""

    ignore_eos: Optional[bool] = False
    """Whether to ignore the EOS token and continue generating tokens after
    the EOS token is generated."""

    min_tokens: Optional[int] = 0
    """Minimum number of tokens to generate per output sequence before 
    EOS or stop_token_ids can be generated"""

    stop_token_ids: Optional[List[int]] = None
    """List of tokens that stop the generation when they are generated.
    The returned output will contain the stop tokens unless the stop tokens
    are special tokens."""

    skip_special_tokens: Optional[bool] = True
    """Whether to skip special tokens in the output. Defaults to True."""

    spaces_between_special_tokens: Optional[bool] = True
    """Whether to add spaces between special tokens in the output.
    Defaults to True."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_model_depolyment_chat_endpoint_vllm"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        params = {
            "model": self.model,
            "stop": self.stop,
            "stream": self.streaming,
        }
        for attr_name in self._get_model_params():
            try:
                value = getattr(self, attr_name)
                if value is not None:
                    params.update({attr_name: value})
            except Exception:
                pass

        return params

    def _get_model_params(self) -> List[str]:
        """Gets the name of model parameters."""
        return [
            "best_of",
            "early_stopping",
            "frequency_penalty",
            "ignore_eos",
            "length_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "min_p",
            "min_tokens",
            "n",
            "presence_penalty",
            "repetition_penalty",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "stop_token_ids",
            "temperature",
            "top_k",
            "top_p",
            "use_beam_search",
        ]


class ChatOCIModelDeploymentTGI(ChatOCIModelDeployment):
    """OCI large language chat models deployed with Text Generation Inference.

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:

        .. code-block:: python

            from langchain_community.chat_models import ChatOCIModelDeploymentTGI

            chat = ChatOCIModelDeploymentTGI(
                endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
                max_token=512,
                temperature=0.2,
                frequency_penalty=0.1,
                seed=42,
                # other model parameters...
            )

    """  # noqa: E501

    frequency_penalty: Optional[float] = None
    """Penalizes repeated tokens according to frequency. Between 0 and 1."""

    logit_bias: Optional[Dict[str, float]] = None
    """Adjust the probability of specific tokens being generated."""

    logprobs: Optional[bool] = None
    """Whether to return log probabilities of the output tokens or not."""

    max_tokens: int = 256
    """The maximum number of tokens to generate in the completion."""

    n: int = 1
    """Number of output sequences to return for the given prompt."""

    presence_penalty: Optional[float] = None
    """Penalizes repeated tokens. Between 0 and 1."""

    seed: Optional[int] = None
    """To sample deterministically,"""

    temperature: float = 0.2
    """What sampling temperature to use."""

    top_p: Optional[float] = None
    """Total probability mass of tokens to consider at each step."""

    top_logprobs: Optional[int] = None
    """An integer between 0 and 5 specifying the number of most 
    likely tokens to return at each token position, each with an 
    associated log probability. logprobs must be set to true if 
    this parameter is used."""

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_model_depolyment_chat_endpoint_tgi"

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        params = {
            "model": self.model,
            "stop": self.stop,
            "stream": self.streaming,
        }
        for attr_name in self._get_model_params():
            try:
                value = getattr(self, attr_name)
                if value is not None:
                    params.update({attr_name: value})
            except Exception:
                pass

        return params

    def _get_model_params(self) -> List[str]:
        """Gets the name of model parameters."""
        return [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "seed",
            "temperature",
            "top_k",
            "top_p",
            "top_logprobs",
        ]
