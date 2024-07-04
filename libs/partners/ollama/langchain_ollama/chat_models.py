"""Ollama chat models."""
from typing import Any, AsyncIterator, Dict, Iterator, List, Literal, Mapping, Optional, Sequence, Union, cast

from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
import ollama
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from ollama import Message, Options, AsyncClient
import asyncio

class ChatOllama(BaseChatModel):
    """Ollama chat model integration.

    Setup:
        Install ``langchain-ollama`` and download any models you want to use from ollama.

        .. code-block:: bash

            ollama pull mistral:v0.3
            pip install -U langchain-ollama

    Key init args — completion params:
        model: str
            Name of Ollama model to use.
        temperature: float
            Sampling temperature. Ranges from 0.0 to 1.0.
        num_predict: Optional[int]
            Max number of tokens to generate.

    See full list of supported init args and their descriptions in the params section.

    Instantiate:
        .. code-block:: python

            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model = "llama3",
                temperature = 0.8,
                num_predict = 256,
                # other params ...
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='J'adore le programmation. (Note: "programming" can also refer to the act of writing code, so if you meant that, I could translate it as "J'adore programmer". But since you didn\'t specify, I assumed you were talking about the activity itself, which is what "le programmation" usually refers to.)', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:37:50.182604Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 3576619666, 'load_duration': 788524916, 'prompt_eval_count': 32, 'prompt_eval_duration': 128125000, 'eval_count': 71, 'eval_duration': 2656556000}, id='run-ba48f958-6402-41a5-b461-5e250a4ebd36-0')

    Stream:
        .. code-block:: python

            messages = [
                ("human", "Return the words Hello World!"),
            ]
            for chunk in llm.stream(messages):
                print(chunk)
            

        .. code-block:: python

            content='Hello' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
            content=' World' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
            content='!' id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'
            content='' response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:39:42.274449Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 411875125, 'load_duration': 1898166, 'prompt_eval_count': 14, 'prompt_eval_duration': 297320000, 'eval_count': 4, 'eval_duration': 111099000} id='run-327ff5ad-45c8-49fe-965c-0a93982e9be1'


        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            AIMessageChunk(content='Je adore le programmation.(Note: "programmation" is the formal way to say "programming" in French, but informally, people might use the phrase "le développement logiciel" or simply "le code")', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:38:54.933154Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 1977300042, 'load_duration': 1345709, 'prompt_eval_duration': 159343000, 'eval_count': 47, 'eval_duration': 1815123000}, id='run-3c81a3ed-3e79-4dd3-a796-04064d804890')

    Async:
        .. code-block:: python

            messages = [
                ("human", "Hello how are you!"),
            ]
            await llm.ainvoke(messages)

        .. code-block:: python

            AIMessage(content="Hi there! I'm just an AI, so I don't have feelings or emotions like humans do. But I'm functioning properly and ready to help with any questions or tasks you may have! How can I assist you today?", response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:52:08.165478Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 2138492875, 'load_duration': 1364000, 'prompt_eval_count': 10, 'prompt_eval_duration': 297081000, 'eval_count': 47, 'eval_duration': 1838524000}, id='run-29c510ae-49a4-4cdd-8f23-b972bfab1c49-0')

        .. code-block:: python

            messages = [
                ("human", "Say hello world!"),
            ]
            async for chunk in llm.astream(messages):
                print(chunk.content)

        .. code-block:: python

            HEL
            LO
            WORLD
            !
        
        .. code-block:: python

            messages = [
                ("human", "Say hello world!"),
                ("human","Say goodbye world!")
            ]
            await llm.abatch(messages)

        .. code-block:: python

            [AIMessage(content='HELLO, WORLD!', response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:55:07.315396Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 1696745458, 'load_duration': 1505000, 'prompt_eval_count': 8, 'prompt_eval_duration': 111627000, 'eval_count': 6, 'eval_duration': 185181000}, id='run-da6c7562-e25a-4a44-987a-2c83cd8c2686-0'),
            AIMessage(content="It's been a blast chatting with you! Say goodbye to the world for me, and don't forget to come back and visit us again soon!", response_metadata={'model': 'llama3', 'created_at': '2024-07-04T03:55:07.018076Z', 'message': {'role': 'assistant', 'content': ''}, 'done_reason': 'stop', 'done': True, 'total_duration': 1399391083, 'load_duration': 1187417, 'prompt_eval_count': 20, 'prompt_eval_duration': 230349000, 'eval_count': 31, 'eval_duration': 1166047000}, id='run-96cad530-6f3e-4cf9-86b4-e0f8abba4cdb-0')]

    JSON mode:
        .. code-block:: python


            json_llm = ChatOllama(format="json")
            messages = [
                ("human", "Return a query for the weather in a random location and time of day with two keys: location and time_of_day. Respond using JSON only."),
            ]
            llm.invoke(messages).content

        .. code-block:: python

            '{"location": "Pune, India", "time_of_day": "morning"}'
    """  # noqa: E501

    model: str = "llama2"
    """Model name to use."""

    mirostat: Optional[int] = None
    """Enable Mirostat sampling for controlling perplexity.
    (default: 0, 0 = disabled, 1 = Mirostat, 2 = Mirostat 2.0)"""

    mirostat_eta: Optional[float] = None
    """Influences how quickly the algorithm responds to feedback
    from the generated text. A lower learning rate will result in
    slower adjustments, while a higher learning rate will make
    the algorithm more responsive. (Default: 0.1)"""

    mirostat_tau: Optional[float] = None
    """Controls the balance between coherence and diversity
    of the output. A lower value will result in more focused and
    coherent text. (Default: 5.0)"""

    num_ctx: Optional[int] = None
    """Sets the size of the context window used to generate the
    next token. (Default: 2048)	"""

    num_gpu: Optional[int] = None
    """The number of GPUs to use. On macOS it defaults to 1 to
    enable metal support, 0 to disable."""

    num_thread: Optional[int] = None
    """Sets the number of threads to use during computation.
    By default, Ollama will detect this for optimal performance.
    It is recommended to set this value to the number of physical
    CPU cores your system has (as opposed to the logical number of cores)."""

    num_predict: Optional[int] = None
    """Maximum number of tokens to predict when generating text.
    (Default: 128, -1 = infinite generation, -2 = fill context)"""

    repeat_last_n: Optional[int] = None
    """Sets how far back for the model to look back to prevent
    repetition. (Default: 64, 0 = disabled, -1 = num_ctx)"""

    repeat_penalty: Optional[float] = None
    """Sets how strongly to penalize repetitions. A higher value (e.g., 1.5)
    will penalize repetitions more strongly, while a lower value (e.g., 0.9)
    will be more lenient. (Default: 1.1)"""

    temperature: Optional[float] = None
    """The temperature of the model. Increasing the temperature will
    make the model answer more creatively. (Default: 0.8)"""

    stop: Optional[Sequence[str]] = None
    """Sets the stop tokens to use."""

    tfs_z: Optional[float] = None
    """Tail free sampling is used to reduce the impact of less probable
    tokens from the output. A higher value (e.g., 2.0) will reduce the
    impact more, while a value of 1.0 disables this setting. (default: 1)"""

    top_k: Optional[int] = None
    """Reduces the probability of generating nonsense. A higher value (e.g. 100)
    will give more diverse answers, while a lower value (e.g. 10)
    will be more conservative. (Default: 40)"""

    top_p: Optional[float] = None
    """Works together with top-k. A higher value (e.g., 0.95) will lead
    to more diverse text, while a lower value (e.g., 0.5) will
    generate more focused and conservative text. (Default: 0.9)"""

    format: Literal['', 'json'] = ''
    """Specify the format of the output (options: json)"""

    keep_alive: Optional[Union[int, str]] = None
    """How long the model will stay loaded into memory."""


    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling Ollama."""
        return {
            "model": self.model,
            "format": self.format,
            "options": {
                "mirostat": self.mirostat,
                "mirostat_eta": self.mirostat_eta,
                "mirostat_tau": self.mirostat_tau,
                "num_ctx": self.num_ctx,
                "num_gpu": self.num_gpu,
                "num_thread": self.num_thread,
                "num_predict": self.num_predict,
                "repeat_last_n": self.repeat_last_n,
                "repeat_penalty": self.repeat_penalty,
                "temperature": self.temperature,
                "stop": self.stop,
                "tfs_z": self.tfs_z,
                "top_k": self.top_k,
                "top_p": self.top_p,
            },
            "keep_alive": self.keep_alive,
        }


    def _convert_messages_to_ollama_messages(
        self, messages: List[BaseMessage]
    ) -> Sequence[Message]:
        ollama_messages: List = []
        for message in messages:
            role = ""
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            elif isinstance(message, SystemMessage):
                role = "system"
            else:
                raise ValueError("Received unsupported message type for Ollama.")

            content = ""
            images = []
            if isinstance(message.content, str):
                content = message.content
            else:
                for content_part in cast(List[Dict], message.content):
                    if content_part.get("type") == "text":
                        content += f"\n{content_part['text']}"
                    elif content_part.get("type") == "image_url":
                        image_url = None
                        temp_image_url = content_part.get("image_url")
                        if isinstance(temp_image_url, str):
                            image_url = content_part["image_url"]
                        elif (
                            isinstance(temp_image_url, dict) and "url" in temp_image_url
                        ):
                            image_url = temp_image_url
                        else:
                            raise ValueError(
                                "Only string image_url or dict with string 'url' "
                                "inside content parts are supported."
                            )

                        image_url_components = image_url.split(",")
                        # Support data:image/jpeg;base64,<image> format
                        # and base64 strings
                        if len(image_url_components) > 1:
                            images.append(image_url_components[1])
                        else:
                            images.append(image_url_components[0])

                    else:
                        raise ValueError(
                            "Unsupported message content type. "
                            "Must either have type 'text' or type 'image_url' "
                            "with a string 'image_url' field."
                        )

            ollama_messages.append(
                {
                    "role": role,
                    "content": content,
                    "images": images,
                }
            )

        return ollama_messages

    async def _acreate_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Union[Mapping[str, Any], str]]:
        ollama_messages = self._convert_messages_to_ollama_messages(messages)
        
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        params["options"]["stop"] = stop
        async for part in await AsyncClient().chat(
            model=params['model'],
            messages=ollama_messages,
            stream=True,
            options=Options(**params['options']),
            keep_alive=params["keep_alive"],
            format=params["format"],
        ):
            yield part

    def _create_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        ollama_messages = self._convert_messages_to_ollama_messages(messages)
        
        if self.stop is not None and stop is not None:
            raise ValueError("`stop` found in both the input and default params.")
        elif self.stop is not None:
            stop = self.stop

        params = self._default_params

        for key in self._default_params:
            if key in kwargs:
                params[key] = kwargs[key]

        params["options"]["stop"] = stop
        yield from ollama.chat(
            model=params['model'],
            messages=ollama_messages,
            stream=True,
            options=Options(**params['options']),
            keep_alive=params["keep_alive"],
            format=params["format"],
        )

    def _chat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=stream_resp["message"]["content"]
                        if "message" in stream_resp
                        else ""
                    ),
                    generation_info=dict(stream_resp)
                    if stream_resp.get("done") is True
                    else None,
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    async def _achat_stream_with_aggregation(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> ChatGenerationChunk:
        final_chunk = None
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=stream_resp["message"]["content"]
                        if "message" in stream_resp
                        else ""
                    ),
                    generation_info=dict(stream_resp)
                    if stream_resp.get("done") is True
                    else None,
                )
                if final_chunk is None:
                    final_chunk = chunk
                else:
                    final_chunk += chunk
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        chunk=chunk,
                        verbose=verbose,
                    )
        if final_chunk is None:
            raise ValueError("No data received from Ollama stream.")

        return final_chunk

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = self._chat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        for stream_resp in self._create_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=stream_resp["message"]["content"]
                        if "message" in stream_resp
                        else ""
                    ),
                    generation_info=dict(stream_resp)
                    if stream_resp.get("done") is True
                    else None,
                )
                if run_manager:
                    run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        async for stream_resp in self._acreate_chat_stream(messages, stop, **kwargs):
            if not isinstance(stream_resp, str):
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=stream_resp["message"]["content"]
                        if "message" in stream_resp
                        else ""
                    ),
                    generation_info=dict(stream_resp)
                    if stream_resp.get("done") is True
                    else None,
                )
                if run_manager:
                    await run_manager.on_llm_new_token(
                        chunk.text,
                        verbose=self.verbose,
                    )
                yield chunk

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        final_chunk = await self._achat_stream_with_aggregation(
            messages, stop, run_manager, verbose=self.verbose, **kwargs
        )
        chat_generation = ChatGeneration(
            message=AIMessage(content=final_chunk.text),
            generation_info=final_chunk.generation_info,
        )
        return ChatResult(generations=[chat_generation])

    # TODO: Work in progress for binding tools
    """ 
    def bind_tools(
        self,
        tools: Sequence[Union[Dict[str, Any], Type[BaseModel], Callable, BaseTool]],
        **kwargs: Any,    
    ):
        if self.model != "mistral:v0.3":
            raise ValueError(f"Binding tools is not available for model {self.model}")
        else:
            tool_prompt = "[AVAILABLE_TOOLS]["
            for tool in tools:
                tool_prompt += json.dumps(convert_to_openai_tool(tool)) + ","
            tool_prompt = tool_prompt[:-1]
            tool_prompt += "][/AVAILABLE_TOOLS]"

            return self.bind(tool_prompt=tool_prompt)

    def with_structured_output(
        self,
        schema: Union[Dict, Type[BaseModel]] = None,
        *,
        method: Literal["function_calling", "json_mode"] = "function_calling",
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, Union[Dict, BaseModel]]:
        if kwargs:
            raise ValueError(f"Received unsupported arguments {kwargs}")
        is_pydantic_schema = _is_pydantic_class(schema)
        if method == "function_calling":
            if schema is None:
                raise ValueError(
                    "schema must be specified when method is 'function_calling'. "
                    "Received None."
                )
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
        elif method == "json_mode":
            llm = self.bind(format="json")
            output_parser = (
                PydanticOutputParser(pydantic_object=schema)
                if is_pydantic_schema
                else JsonOutputParser()
            )
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
            return llm #| output_parser
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-ollama"
