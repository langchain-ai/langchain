"""Ollama chat models."""
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Union, cast

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
from ollama import Message, Options


class ChatOllama(BaseChatModel):
    """Ollama chat model integration.

    # TODO: Replace with relevant packages, env vars.
    Setup:
        Install ``langchain-ollama`` and set environment variable ``OLLAMA_API_KEY``.

        .. code-block:: bash

            pip install -U langchain-ollama
            export OLLAMA_API_KEY="your-api-key"

    # TODO: Populate with relevant params.
    Key init args — completion params:
        model: str
            Name of Ollama model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    # TODO: Populate with relevant params.
    Key init args — client params:
        timeout: Optional[float]
            Timeout for requests.
        max_retries: int
            Max number of retries.
        api_key: Optional[str]
            Ollama API key. If not passed in will be read from env var OLLAMA_API_KEY.

    See full list of supported init args and their descriptions in the params section.

    # TODO: Replace with relevant init params.
    Instantiate:
        .. code-block:: python

            from langchain_ollama import ChatOllama

            llm = ChatOllama(
                model="...",
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
                ("system", "You are a helpful translator. Translate the user sentence to French."),
                ("human", "I love programming."),
            ]
            llm.invoke(messages)

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if token-level streaming isn't supported.
    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk)

        .. code-block:: python

            # TODO: Example output.

        .. code-block:: python

            stream = llm.stream(messages)
            full = next(stream)
            for chunk in stream:
                full += chunk
            full

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if native async isn't supported.
    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

            # stream:
            # async for chunk in (await llm.astream(messages))

            # batch:
            # await llm.abatch([messages])

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if JSON mode response format isn't supported.
    JSON mode:
        .. code-block:: python

            # TODO: Replace with appropriate bind arg.
            json_llm = llm.bind(response_format={"type": "json_object"})
            ai_msg = json_llm.invoke("Return a JSON object with key 'random_ints' and a value of 10 random ints in [0-99]")
            ai_msg.content

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if image inputs aren't supported.
    Image input:
        .. code-block:: python

            import base64
            import httpx
            from langchain_core.messages import HumanMessage

            image_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
            image_data = base64.b64encode(httpx.get(image_url).content).decode("utf-8")
            # TODO: Replace with appropriate message content format.
            message = HumanMessage(
                content=[
                    {"type": "text", "text": "describe the weather in this image"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
                    },
                ],
            )
            ai_msg = llm.invoke([message])
            ai_msg.content

        .. code-block:: python

            # TODO: Example output.

    # TODO: Delete if audio inputs aren't supported.
    Audio input:
        .. code-block:: python

            # TODO: Example input

        .. code-block:: python

            # TODO: Example output

    # TODO: Delete if video inputs aren't supported.
    Video input:
        .. code-block:: python

            # TODO: Example input

        .. code-block:: python

            # TODO: Example output

    # TODO: Delete if token usage metadata isn't supported.
    Token usage:
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.usage_metadata

        .. code-block:: python

            {'input_tokens': 28, 'output_tokens': 5, 'total_tokens': 33}

    # TODO: Delete if logprobs aren't supported.
    Logprobs:
        .. code-block:: python

            # TODO: Replace with appropriate bind arg.
            logprobs_llm = llm.bind(logprobs=True)
            ai_msg = logprobs_llm.invoke(messages)
            ai_msg.response_metadata["logprobs"]

        .. code-block:: python

              # TODO: Example output.

    Response metadata
        .. code-block:: python

            ai_msg = llm.invoke(messages)
            ai_msg.response_metadata

        .. code-block:: python

             # TODO: Example output.

    """  # noqa: E501

    model: str = "llama2"
    """Model name to use."""
    tool_prompt: Optional[str] = None

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

    def _create_chat_stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[Union[Mapping[str, Any], str]]:
        ollama_messages = self._convert_messages_to_ollama_messages(messages)
        """
        if 'tool_prompt' in kwargs:
            ollama_messages[-1]['content'] = kwargs['tool_prompt'] + "[INST]" + ollama_messages[-1]['content'] + "[/INST]"
        """  # noqa: E501
        options_data: Dict = {
            k: v
            for k, v in kwargs.items()
            if k not in ["keep_alive", "format"] and k in Options.__annotations__
        }
        options_data["stop"] = stop
        yield from ollama.chat(
            model=self.model,
            messages=ollama_messages,
            stream=True,
            options=Options(**options_data),
            keep_alive=kwargs.get("keep_alive", None),
            format=kwargs.get("format", None),
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

    # TODO: Implement if ChatOllama supports async streaming. Otherwise delete.
    # async def _astream(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> AsyncIterator[ChatGenerationChunk]:

    # TODO: Implement if ChatOllama supports async generation. Otherwise delete.
    # async def _agenerate(
    #     self,
    #     messages: List[BaseMessage],
    #     stop: Optional[List[str]] = None,
    #     run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
    #     **kwargs: Any,
    # ) -> ChatResult:

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
