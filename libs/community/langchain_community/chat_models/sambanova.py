import json
from typing import Any, Dict, Iterator, List, Optional, Tuple

import requests
from requests import Response
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
    ChatMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from pydantic import Field, SecretStr

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

def _create_message_dicts(
    messages: List[BaseMessage]
) -> List[Dict[str, Any]]:
    """
    convert a list of BaseMessages to a list of dictionaries with Role / content

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
        To use, you should have the environment variables
        ``SAMBANOVA_URL`` set with your SambaNova Cloud URL.
        ``SAMBANOVA_API_KEY`` set with your SambaNova Cloud API Key.
        http://cloud.sambanova.ai/
        Example:
        .. code-block:: python
            ChatSambaNovaCloud(
                sambanova_url = SambaNova cloud endpoint URL,
                sambanova_api_key = set with your SambaNova cloud API key,
                model = model name,
                streaming = set True for use streaming API
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                stream_options = include usage to get generation metrics
            )

    Key init args — completion params:
        model: str
            The name of the model to use, e.g., llama3-8b.
        streaming: bool
            Whether to use streaming or not
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
                streaming = set True for streaming
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
    """Whether to use streaming or not"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: float = Field(default=0.7)
    """model temperature"""

    top_p: float = Field(default=0.0)
    """model top p"""

    top_k: int = Field(default=1)
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
                f"{response.status_code}."
                f"{response.text}."
            )
        response_dict = response.json()
        if response_dict.get("error"):
            raise RuntimeError(
                f"Sambanova /complete call failed with status code "
                f"{response.status_code}."
                f"{response_dict}."
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

        Returns:
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
            chunk = {
                "event": event.event,
                "data": event.data,
                "status_code": response.status_code,
            }

            if chunk["event"] == "error_event" or chunk["status_code"] != 200:
                raise RuntimeError(
                    f"Sambanova /complete call failed with status code "
                    f"{chunk['status_code']}."
                    f"{chunk}."
                )

            try:
                # check if the response is a final event
                # in that case event data response is '[DONE]'
                if chunk["data"] != "[DONE]":
                    if isinstance(chunk["data"], str):
                        data = json.loads(chunk["data"])
                    else:
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{chunk['status_code']}."
                            f"{chunk}."
                        )
                    if data.get("error"):
                        raise RuntimeError(
                            f"Sambanova /complete call failed with status code "
                            f"{chunk['status_code']}."
                            f"{chunk}."
                        )
                    yield data
            except Exception:
                raise Exception(
                    f"Error getting content chunk raw streamed response: {chunk}"
                )

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        SambaNovaCloud chat model logic.

        Call SambaNovaCloud API.

        Args:
            messages: the prompt composed of a list of messages.
            stop: a list of strings on which the model should stop generating.
                  If generation stops due to a stop token, the stop token itself
                  SHOULD BE INCLUDED as part of the output. This is not enforced
                  across models right now, but it's a good practice to follow since
                  it makes it much easier to parse the output of the model
                  downstream and understand why generation stopped.
            run_manager: A run manager with callbacks for the LLM.
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
        To use, you should have the environment variables
        ``SAMBASTUDIO_URL`` set with your SambaStudio deployed endpoint URL.
        ``SAMBASTUDIO_API_KEY`` set with your SambaStudio deployed endpoint Key.
        https://docs.sambanova.ai/sambastudio/latest/index.html
        Example:
        .. code-block:: python
            ChatSambaStudio(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model name,
                streaming = set True for use streaming API
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                stream_options = include usage to get generation metrics
                model_kwargs: Optional = Extra Key word arguments to pass to the model.
            )

    Key init args — completion params:
        model: str
            The name of the model to use, e.g., llama3-8b.
        streaming: bool
            Whether to use streaming or not
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

            ChatSambaStudio(
                sambastudio_url = set with your SambaStudio deployed endpoint URL,
                sambastudio_api_key = set with your SambaStudio deployed endpoint Key.
                model = model name,
                streaming = set True for streaming
                max_tokens = max number of tokens to generate,
                temperature = model temperature,
                top_p = model top p,
                top_k = model top k,
                process_prompt = wether to process prompt or not (for generic api v1 and v2),
                do_sample = wether to do sample or not (for generic api v1 and v2),
                stream_options = include usage to get generation metrics (for openai api)
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

    model: str = Field(default="Meta-Llama-3.1-8B-Instruct")
    """The name of the model or expert to use"""

    streaming: bool = Field(default=False)
    """Whether to use streaming api or not when using invoke method"""

    max_tokens: int = Field(default=1024)
    """max tokens to generate"""

    temperature: float = Field(default=0.7)
    """model temperature"""

    top_p: float = Field()
    """model top p"""

    top_k: int = Field()
    """model top k"""
    
    do_sample: bool = Field(default=True)
    """whether to do sampling"""
    
    process_prompt: bool = Field(default=False)
    """whether process prompt"""

    stream_options: dict = Field(default={"include_usage": True})
    """stream options, include usage to get generation metrics"""
    
    special_tokens: dict = Field(
        default={
        "start" : "<|begin_of_text|>",
        "start_role" : "<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>",
        "end_role" : "<|eot_id|>",
        "end" : "<|start_header_id|>assistant<|end_header_id|>\n"
        }
    )
    
    model_kwargs: Optional[dict] = None
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
            "sambastudio_api_key": "sambastudio_api_key"
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
            "stream_options": self.stream_options,
            "model_kwargs": self.model_kwargs
        }

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "sambastudio-chatmodel"

    def __init__(self, **kwargs: Any) -> None:
            """init and validate environment variables"""
            kwargs["sambastudio_url"] = get_from_dict_or_env(kwargs, "sambastudio_url", "SAMBASTUDIO_URL")
            
            kwargs["sambastudio_api_key"] = convert_to_secret_str(
                get_from_dict_or_env(kwargs, "sambastudio_api_key", "SAMBASTUDIO_API_KEY")
            )
            kwargs["base_url"], kwargs["streaming_url"] = self._get_sambastudio_urls(
                kwargs["sambastudio_url"]
            )
            super().__init__(**kwargs)
     
    def _get_role(self, message:BaseMessage) -> str:
        """
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
            role =  "tool"
        else:
            raise TypeError(f"Got unknown type {message}")
        return role
    def _messages_to_string(self, messages: List[BaseMessage]) -> str:
        """
        """ 
        if self.process_prompt:
            messages_dict={
                "conversation_id":"sambaverse-conversation-id",
                "messages":[]
                }
            for message in messages:
                
                messages_dict["messages"].append(
                    {
                        "message_id": message.id,
                        "role": self._get_role(message),
                        "content": message.content
                    }
                )
            messages_string = json.dumps(messages_dict)
        else:
            messages_string = self.special_tokens["start"]
            for message in messages:
                messages_string += self.special_tokens['start_role'].format(role=self._get_role(message))
                messages_string += f' {message.content} '
                messages_string += self.special_tokens["end_role"]
            messages_string += self.special_tokens["end"]
               
        return messages_string
            
    def _get_sambastudio_urls(self, url: str) -> Tuple[str, str]:
        """
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
        streaming: Optional[bool] = False
    ) -> Dict[str, Any]:
        
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
                "streaming": streaming,
            }
            data = {key:value for key, value in data.items() if value is not None}
            headers = {
                "Authorization": f"Bearer {self.sambastudio_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            }
        
        # create request payload for generic v1 API
        elif "api/v2/predict/generic" in self.sambastudio_url:
            items = [{"id": "item0", "value": self._messages_to_string(messages)}]
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
            params = {key:value for key, value in params.items() if value is not None}
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
            params = {key:{"type":type(value).__name__, "value": str(value)} for key, value in params.items() if value is not None}
            data = {
                "instances": [self._messages_to_string(messages)],
                "params": params
                }
            headers = {"key": self.sambastudio_api_key.get_secret_value()}   
            
        else:
            raise ValueError(f"Unsupported URL{self.sambastudio_url}",
                             "only openai, generic v1 and generic v2 APIs are supported")
        
             
        http_session = requests.Session()
        response = http_session.post(
                self.base_url,
                headers=headers,
                json=data,
                stream = streaming
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
        """
        
        #Extract json payload form response
        try:
            response_dict = response.json()
        except:
            raise RuntimeError(
                f"Sambanova /complete call failed couldn't get JSON response"
                f"{response.text}"
            )
        
        # process response payload for openai compatible API
        if "openai" in self.sambastudio_url:
            content=response_dict["choices"][0]["message"]["content"]
            id=response_dict["id"]
            response_metadata={
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
            id = ""
            response_metadata = response_dict
        
        else:
            raise ValueError(f"Unsupported URL{self.sambastudio_url}",
                             "only openai, generic v1 and generic v2 APIs are supported")
            
        return AIMessage(
                content=content,
                additional_kwargs={},
                response_metadata=response_metadata,
                id=id,
            )
        
    def _process_stream_response(self, response: Response):
        """
        """
        
        # process response payload for openai compatible API
        if "openai" in self.sambastudio_url:
            pass  
            
        # process response payload for generic v2 API
        elif "api/v2/predict/generic" in self.sambastudio_url:   
            pass
        
        # process response payload for generic v1 API
        elif "api/predict/generic" in self.sambastudio_url:   
            pass
        
        content = ""
        response_metadata = {}
        id = ""
        return ChatGenerationChunk(
                    message=AIMessageChunk(
                        content=content,
                        id=id,
                        response_metadata=response_metadata,
                        additional_kwargs={},
                    )
                )
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
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
        """
        response = self._handle_request(messages, stop, streaming=True) 
        for chunk in self._process_stream_response(response): #TODO process stream response method method
            if run_manager:
                run_manager.on_llm_new_token(chunk.text, chunk=chunk)
            yield chunk