import json
import logging
from operator import itemgetter
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Optional,
    Type,
    Union,
)

import aiohttp
import requests
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
from langchain_core.language_models.llms import create_base_retry_decorator
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.pydantic_v1 import BaseModel, Field, root_validator
from langchain_core.runnables import Runnable, RunnableMap, RunnablePassthrough
from langchain_core.utils import (
    get_from_dict_or_env,
)

from langchain_community.adapters.openai import (
    convert_dict_to_message,
    convert_message_to_dict,
)
from langchain_community.chat_models.openai import _convert_delta_to_message_chunk
from langchain_community.utilities.requests import Requests

logger = logging.getLogger(__name__)

DEFAULT_TIME_OUT = 300
DEFAULT_CONTENT_TYPE_JSON = "application/json"
DEFAULT_MODEL_NAME = "odsc-llm"


def _is_pydantic_class(obj: Any) -> bool:
    return isinstance(obj, type) and issubclass(obj, BaseModel)


class TokenExpiredError(Exception):
    pass


class ServerError(Exception):
    pass


def _create_retry_decorator(
    llm,
    *,
    run_manager: Optional[
        Union[AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun]
    ] = None,
) -> Callable[[Any], Any]:
    """Create a retry decorator."""
    errors = [requests.exceptions.ConnectTimeout, TokenExpiredError]
    decorator = create_base_retry_decorator(
        error_types=errors, max_retries=llm.max_retries, run_manager=run_manager
    )
    return decorator


class ChatOCIModelDeploymentEndpoint(BaseChatModel):
    """OCI Data Science Model Deployment chat model integration.

    Instantiate:
        .. code-block:: python

            from langchain_community.chat_models import ChatOCIModelDeploymentEndpoint

            chat = ChatOCIModelDeploymentEndpoint(
                endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict",
                model="odsc-llm",
                # other params...
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

        See ``ChatOCIModelDeploymentEndpoint.with_structured_output()`` for more.

    """  # noqa: E501

    auth: dict = Field(default_factory=dict, exclude=True)
    """ADS auth dictionary for OCI authentication:
    https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html.
    This can be generated by calling `ads.common.auth.api_keys()`
    or `ads.common.auth.resource_principal()`. If this is not
    provided then the `ads.common.default_signer()` will be used."""

    endpoint: str = ""
    """The uri of the endpoint from the deployed Model Deployment model."""

    streaming: bool = False
    """Whether to stream the results or not."""

    model_kwargs: Dict[str, Any] = Field(default_factory=dict)
    """Keyword arguments to pass to the model."""

    model: str = DEFAULT_MODEL_NAME
    """The name of the model."""

    max_retries: int = 3
    """Maximum number of retries to make when generating."""

    stop: Optional[List[str]] = None
    """Stop words to use when generating. Model output is cut off
    at the first occurrence of any of these substrings."""

    @root_validator()
    def validate_environment(  # pylint: disable=no-self-argument
        cls, values: Dict
    ) -> Dict:
        """Validate that python package exists in environment."""
        try:
            import ads

        except ImportError as ex:
            raise ImportError(
                "Could not import ads python package. "
                "Please install it with `pip install oracle_ads`."
            ) from ex
        if not values.get("auth", None):
            values["auth"] = ads.common.auth.default_signer()
        values["endpoint"] = get_from_dict_or_env(
            values,
            "endpoint",
            "OCI_LLM_ENDPOINT",
        )
        return values

    @property
    def _llm_type(self) -> str:
        """Return type of llm."""
        return "oci_model_depolyment_chat_endpoint"

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this model can be serialized by Langchain."""
        return True

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            **{"endpoint": self.endpoint},
            **self._default_params,
        }

    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters."""
        return {
            "model": self.model,
            "stop": self.stop,
            "stream": self.streaming,
            **self.model_kwargs,
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Call out to an OCI Model Deployment Online endpoint.

        Args:
            messages:  The messages in the conversation with the chat model.
            stop: Optional list of stop words to use when generating.

        Returns:
            LangChain ChatResult

        Raises:
            ValueError:
                Raise when invoking endpoint fails.

        Example:

            .. code-block:: python

                from langchain_core.messages import HumanMessage, AIMessage

                messages = [
                            HumanMessage(content="hello!"),
                            AIMessage(content="Hi there human!"),
                            HumanMessage(content="Meow!")
                          ]
                response = chat.invoke(messages)
        """
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
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
            ValueError:
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
        stream: Optional[bool] = None,
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
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
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
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
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
        params["stop"] = stop or params["stop"]
        return {**params, **kwargs}

    def _headers(self, is_async=False, body=None) -> Dict:
        """Construct and return the headers for a request.

        Args:
            is_async (bool, optional): Indicates if the request is asynchronous.
                Defaults to `False`.
            body (optional): The request body to be included in the headers if
                the request is asynchronous.

        Returns:
            Dict: A dictionary containing the appropriate headers for the request.
        """
        if is_async:
            signer = self.auth["signer"]
            req = requests.Request("POST", self.endpoint, json=body)
            req = req.prepare()
            req = signer(req)
            headers = {}
            for key, value in req.headers.items():
                headers[key] = value

            if self.streaming:
                headers.update(
                    {"enable-streaming": "true", "Accept": "text/event-stream"}
                )
            return headers

        return (
            {
                "Content-Type": DEFAULT_CONTENT_TYPE_JSON,
                "enable-streaming": "true",
                "Accept": "text/event-stream",
            }
            if self.streaming
            else {
                "Content-Type": DEFAULT_CONTENT_TYPE_JSON,
            }
        )

    def completion_with_retry(
        self, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any
    ) -> Any:
        """Use tenacity to retry the completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        def _completion_with_retry(**kwargs: Any) -> Any:
            try:
                request_timeout = kwargs.pop("request_timeout", DEFAULT_TIME_OUT)
                data = kwargs.pop("data")
                stream = kwargs.pop("stream", self.streaming)

                request = Requests(
                    headers=self._headers(), auth=self.auth.get("signer")
                )
                response = request.post(
                    url=self.endpoint,
                    data=data,
                    timeout=request_timeout,
                    stream=stream,
                    **kwargs,
                )
                self._check_response(response)
                return response
            except TokenExpiredError as e:
                raise e
            except Exception as err:
                logger.debug(
                    f"Requests payload: {data}. Requests arguments: "
                    f"url={self.endpoint},timeout={request_timeout},stream={stream}."
                    f"Additional request kwargs={kwargs}."
                )
                raise ValueError(
                    f"Error occurs by inference endpoint: {str(err)}"
                ) from err

        return _completion_with_retry(**kwargs)

    async def acompletion_with_retry(
        self,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Use tenacity to retry the async completion call."""
        retry_decorator = _create_retry_decorator(self, run_manager=run_manager)

        @retry_decorator
        async def _completion_with_retry(**kwargs: Any) -> Any:
            try:
                request_timeout = kwargs.pop("request_timeout", DEFAULT_TIME_OUT)
                data = kwargs.pop("data")
                stream = kwargs.pop("stream", self.streaming)

                request = Requests(headers=self._headers(is_async=True, body=data))
                if stream:
                    response = request.apost(
                        url=self.endpoint,
                        data=data,
                        timeout=request_timeout,
                    )
                    return self._aiter_sse(response)
                else:
                    async with request.apost(
                        url=self.endpoint,
                        data=data,
                        timeout=request_timeout,
                    ) as response:
                        self._check_response(response)
                        data = await response.json()
                        return data
            except TokenExpiredError as e:
                raise e
            except Exception as err:
                logger.debug(
                    f"Requests payload: `{data}`. "
                    f"Stream mode={stream}. "
                    f"Requests kwargs: url={self.endpoint}, timeout={request_timeout}."
                )
                raise ValueError(
                    f"Error occurs by inference endpoint: {str(err)}"
                ) from err

        return await _completion_with_retry(**kwargs)

    def _check_response(
        self, response: Union[requests.Response, aiohttp.ClientResponse]
    ) -> None:
        """Handle server error by checking the response status.

        Args:
            response (Union[requests.Response, aiohttp.ClientResponse]):
                The response object from either `requests` or `aiohttp` library.

        Raises:
            TokenExpiredError:
                If the response status code is 401 and the token refresh is successful.
            ServerError:
                If any other HTTP error occurs.
        """
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as http_err:
            status_code = (
                response.status_code
                if hasattr(response, "status_code")
                else response.status
            )
            if status_code == 401 and self._refresh_signer():
                raise TokenExpiredError() from http_err
            else:
                raise ServerError(
                    f"Server error: {str(http_err)}. \nMessage: {response.text}"
                ) from http_err

    def _parse_stream(self, lines: Iterator[bytes]) -> Iterator[str]:
        """Parse a stream of byte lines and yield parsed string lines.

        Args:
            lines (Iterator[bytes]):
                An iterator that yields lines in byte format.

        Yields:
            Iterator[str]:
                An iterator that yields parsed lines as strings.
        """
        for line in lines:
            _line = self._parse_stream_line(line)
            if _line is not None:
                yield _line

    async def _parse_stream_async(
        self,
        lines: aiohttp.StreamReader,
    ) -> AsyncIterator[str]:
        """
        Asynchronously parse a stream of byte lines and yield parsed string lines.

        Args:
            lines (aiohttp.StreamReader):
                An `aiohttp.StreamReader` object that yields lines in byte format.

        Yields:
            AsyncIterator[str]:
                An asynchronous iterator that yields parsed lines as strings.
        """
        async for line in lines:
            _line = self._parse_stream_line(line)
            if _line is not None:
                yield _line

    def _parse_stream_line(self, line: bytes) -> Optional[str]:
        """Parse a single byte line and return a processed string line if valid.

        Args:
            line (bytes): A single line in byte format.

        Returns:
            Optional[str]:
                The processed line as a string if valid, otherwise `None`.
        """
        line = line.strip()
        if line:
            line = line.decode("utf-8")
            if "[DONE]" in line:
                return None

            if line.startswith("data:"):
                return line[5:].lstrip()
        return None

    def _handle_sse_line(
        self, line: str, default_chunk_cls: AIMessageChunk
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
        except Exception:
            return ChatGenerationChunk()

    async def _aiter_sse(
        self,
        async_cntx_mgr,
    ) -> AsyncIterator[Dict]:
        """Asynchronously iterate over server-sent events (SSE).

        Args:
            async_cntx_mgr: An asynchronous context manager that yields a client
                response object.

        Yields:
            AsyncIterator[Dict]: An asynchronous iterator that yields parsed server-sent
                event lines as dictionaries.
        """
        async with async_cntx_mgr as client_resp:
            self._check_response(client_resp)
            async for line in self._parse_stream_async(client_resp.content):
                yield line

    def _refresh_signer(self) -> None:
        """Attempt to refresh the security token using the signer.

        Returns:
                bool: `True` if the token was successfully refreshed, `False` otherwise.
        """
        if self.auth.get("signer", None) and hasattr(
            self.auth["signer"], "refresh_security_token"
        ):
            self.auth["signer"].refresh_security_token()
            return True
        return False

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
        self, response_json: dict, default_chunk_cls=AIMessageChunk
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


class ChatOCIModelDeploymentEndpointVLLM(ChatOCIModelDeploymentEndpoint):
    """OCI large language chat models deployed with vLLM.

    To use, you must provide the model HTTP endpoint from your deployed
    model, e.g. https://<MD_OCID>/predict.

    To authenticate, `oracle-ads` has been used to automatically load
    credentials: https://accelerated-data-science.readthedocs.io/en/latest/user_guide/cli/authentication.html

    Make sure to have the required policies to access the OCI Data
    Science Model Deployment endpoint. See:
    https://docs.oracle.com/en-us/iaas/data-science/using/model-dep-policies-auth.htm#model_dep_policies_auth__predict-endpoint

    Example:

        .. code-block:: python

            from langchain_community.chat_models import ChatOCIModelDeploymentEndpointVLLM

            oci_md = ChatOCIModelDeploymentEndpointVLLM(
                endpoint="https://modeldeployment.us-ashburn-1.oci.customer-oci.com/<ocid>/predict"
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

        params.update(**self.model_kwargs)
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
