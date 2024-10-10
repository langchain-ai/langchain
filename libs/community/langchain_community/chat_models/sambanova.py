import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from langchain_core.callbacks import (
    CallbackManagerForLLMRun,
)
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
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr
from requests import Response


def _convert_message_to_dict(message: BaseMessage) -> Dict[str, Any]:
    """
    convert a BaseMessage to a dictionary with Role / content

    Args:
        message: BaseMessage

    Returns:
        messages_dict:  role / content dict
    """
    if isinstance(message, ChatMessage):
        message_dict = {"role": message.role, "content": message.content}
    elif isinstance(message, SystemMessage):
        message_dict = {"role": "system", "content": message.content}
    elif isinstance(message, HumanMessage):
        message_dict = {"role": "user", "content": message.content}
    elif isinstance(message, AIMessage):
        message_dict = {"role": "assistant", "content": message.content}
    elif isinstance(message, ToolMessage):
        message_dict = {"role": "tool", "content": message.content}
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


class ChatSambaNovaCloud(BaseChatModel):
    """
    SambaNova Cloud chat model.

    Setup:
        To use, you should have the environment variables:
        ``SAMBANOVA_URL`` set with your SambaNova Cloud URL.
        ``SAMBANOVA_API_KEY`` set with your SambaNova Cloud API Key.
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

    sambanova_api_key: SecretStr = Field(default="")
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

    stream_options: dict = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""

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

    def _handle_request(
        self, messages_dicts: List[Dict], stop: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Performs a post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens

        Returns:
            An iterator of response dicts.
        """
        data = {
            "messages": messages_dicts,
            "max_tokens": self.max_tokens,
            "stop": stop,
            "model": self.model,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }
        http_session = requests.Session()
        response = http_session.post(
            self.sambanova_url,
            headers={
                "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json=data,
        )
        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}.",
                f"{response.text}.",
            )
        response_dict = response.json()
        if response_dict.get("error"):
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}.",
                f"{response_dict}.",
            )
        return response_dict

    def _handle_streaming_request(
        self, messages_dicts: List[Dict], stop: Optional[List[str]] = None
    ) -> Iterator[Dict]:
        """
        Performs an streaming post request to the LLM API.

        Args:
            messages_dicts: List of role / content dicts to use as input.
            stop: list of stop tokens

        Yields:
            An iterator of response dicts.
        """
        try:
            import sseclient
        except ImportError:
            raise ImportError(
                "could not import sseclient library"
                "Please install it with `pip install sseclient-py`."
            )
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
        }
        http_session = requests.Session()
        response = http_session.post(
            self.sambanova_url,
            headers={
                "Authorization": f"Bearer {self.sambanova_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            json=data,
            stream=True,
        )

        client = sseclient.SSEClient(response)

        if response.status_code != 200:
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}."
                f"{response.text}."
            )

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
                    yield data
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
        response = self._handle_request(messages_dicts, stop)
        message = AIMessage(
            content=response["choices"][0]["message"]["content"],
            additional_kwargs={},
            response_metadata={
                "finish_reason": response["choices"][0]["finish_reason"],
                "usage": response.get("usage"),
                "model_name": response["model"],
                "system_fingerprint": response["system_fingerprint"],
                "created": response["created"],
            },
            id=response["id"],
        )

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
        finish_reason = None
        for partial_response in self._handle_streaming_request(messages_dicts, stop):
            if len(partial_response["choices"]) > 0:
                finish_reason = partial_response["choices"][0].get("finish_reason")
                content = partial_response["choices"][0]["delta"]["content"]
                id = partial_response["id"]
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=content, id=id, additional_kwargs={})
                )
            else:
                content = ""
                id = partial_response["id"]
                metadata = {
                    "finish_reason": finish_reason,
                    "usage": partial_response.get("usage"),
                    "model_name": partial_response["model"],
                    "system_fingerprint": partial_response["system_fingerprint"],
                    "created": partial_response["created"],
                }
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        id=id,
                        response_metadata=metadata,
                        additional_kwargs={},
                    )
                )

            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk


class ChatSambaStudio(BaseChatModel):
    """
    SambaStudio chat model.

    Setup:
        To use, you should have the environment variables:
        ``SAMBASTUDIO_URL`` set with your SambaStudio deployed endpoint URL.
        ``SAMBASTUDIO_API_KEY`` set with your SambaStudio deployed endpoint Key.
        https://docs.sambanova.ai/sambastudio/latest/index.html
        Example:
        .. code-block:: python
            ChatSambaStudio(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model or expert name (set for CoE endpoints),
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                do_sample = wether to do sample
                process_prompt = wether to process prompt
                    (set for CoE generic v1 and v2 endpoints)
                stream_options = include usage to get generation metrics
                special_tokens = start, start_role, end_role, end special tokens
                    (set for CoE generic v1 and v2 endpoints when process prompt
                     set to false or for StandAlone v1 and v2 endpoints)
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Key init args — completion params:
        model: str
            The name of the model to use, e.g., Meta-Llama-3-70B-Instruct-4096
            (set for CoE endpoints).
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
            wether to process prompt (set for CoE generic v1 and v2 endpoints)
        stream_options: dict
            stream options, include usage to get generation metrics
        special_tokens: dict
            start, start_role, end_role and end special tokens
            (set for CoE generic v1 and v2 endpoints when process prompt set to false
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
                model = model or expert name (set for CoE endpoints),
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                do_sample = wether to do sample
                process_prompt = wether to process prompt
                    (set for CoE generic v1 and v2 endpoints)
                stream_options = include usage to get generation metrics
                special_tokens = start, start_role, end_role, and special tokens
                    (set for CoE generic v1 and v2 endpoints when process prompt
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

    sambastudio_api_key: SecretStr = Field(default="")
    """SambaStudio api key"""

    base_url: str = Field(default="", exclude=True)
    """SambaStudio non streaming Url"""

    streaming_url: str = Field(default="", exclude=True)
    """SambaStudio streaming Url"""

    model: Optional[str] = Field(default=None)
    """The name of the model or expert to use (for CoE endpoints)"""

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
    """whether process prompt (for CoE generic v1 and v2 endpoints)"""

    stream_options: dict = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""

    special_tokens: dict = Field(
        default={
            "start": "<|begin_of_text|>",
            "start_role": "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>",
            "end_role": "<|eot_id|>",
            "end": "<|start_header_id|>assistant<|end_header_id|>\n",
        }
    )
    """start, start_role, end_role and end special tokens 
    (set for CoE generic v1 and v2 endpoints when process prompt set to false 
     or for StandAlone v1 and v2 endpoints) 
    default to llama3 special tokens"""

    model_kwargs: Optional[Dict[str, Any]] = None
    """Key word arguments to pass to the model."""

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

    def _get_role(self, message: BaseMessage) -> str:
        """
        Get the role of LangChain BaseMessage

        Args:
            message: LangChain BaseMessage

        Returns:
            str: Role of the LangChain BaseMessage
        """
        if isinstance(message, ChatMessage):
            role = message.role
        elif isinstance(message, SystemMessage):
            role = "system"
        elif isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        elif isinstance(message, ToolMessage):
            role = "tool"
        else:
            raise TypeError(f"Got unknown type {message}")
        return role

    def _messages_to_string(self, messages: List[BaseMessage]) -> str:
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
            }
            for message in messages:
                messages_dict["messages"].append(
                    {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content,
                    }
                )
            messages_string = json.dumps(messages_dict)
        else:
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
        if "openai" in url:
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
        if "openai" in self.sambastudio_url:
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
            }
            data = {key: value for key, value in data.items() if value is not None}
            headers = {
                "Authorization": f"Bearer "
                f"{self.sambastudio_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }

        # create request payload for generic v1 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            items = [{"id": "item0", "value": self._messages_to_string(messages)}]
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
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

        # create request payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:
            params = {
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
            headers = {"key": self.sambastudio_api_key.get_secret_value()}

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

        # process response payload for openai compatible API
        if "openai" in self.sambastudio_url:
            content = response_dict["choices"][0]["message"]["content"]
            id = response_dict["id"]
            response_metadata = {
                "finish_reason": response_dict["choices"][0]["finish_reason"],
                "usage": response_dict.get("usage"),
                "model_name": response_dict["model"],
                "system_fingerprint": response_dict["system_fingerprint"],
                "created": response_dict["created"],
            }

        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            content = response_dict["items"][0]["value"]["completion"]
            id = response_dict["items"][0]["id"]
            response_metadata = response_dict["items"][0]

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
            additional_kwargs={},
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
        if "openai" in self.sambastudio_url:
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
        response = self._handle_request(messages, stop, streaming=False)
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
        response = self._handle_request(messages, stop, streaming=True)
        for ai_message_chunk in self._process_stream_response(response):
            chunk = ChatGenerationChunk(message=ai_message_chunk)
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk
