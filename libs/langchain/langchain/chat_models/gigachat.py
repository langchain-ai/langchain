# noqa: I001
"""
GigaChatModel for GigaChat.
"""
from __future__ import annotations

import json
import logging
import os
from typing import (
    Any,
    AsyncIterator,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
)

import requests
import urllib3.exceptions
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt

from langchain.callbacks.manager import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain.chat_models.base import BaseChatModel
from langchain.schema import ChatResult
from langchain.schema.messages import (
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
)
from langchain.schema.output import (
    ChatGeneration,
    ChatGenerationChunk,
)

LATEST_MODEL = "GigaChat:latest"


def parse_stream_helper(line: bytes) -> Optional[str]:
    if line:
        if line.strip() == b"data: [DONE]":
            # return here will cause GeneratorExit exception in urllib3
            # and it will close http connection with TCP Reset
            return None
        if line.startswith(b"data: "):
            line = line[len(b"data: ") :]
            return line.decode("utf-8")
        else:
            return None
    return None


def parse_stream(rbody: Iterator[bytes]) -> Iterator[str]:
    for line in rbody:
        _line = parse_stream_helper(line)
        if _line is not None:
            yield _line


def _convert_delta_to_message_chunk(
    _dict: Mapping[str, Any], default_class: type[BaseMessageChunk]
) -> BaseMessageChunk:
    role = _dict.get("role")
    content = _dict.get("content") or ""
    if _dict.get("function_call"):
        additional_kwargs = {"function_call": dict(_dict["function_call"])}
    else:
        additional_kwargs = {}

    if role == "user" or default_class == HumanMessageChunk:
        return HumanMessageChunk(content=content)
    elif role == "assistant" or default_class == AIMessageChunk:
        return AIMessageChunk(content=content, additional_kwargs=additional_kwargs)
    elif role == "system" or default_class == SystemMessageChunk:
        return SystemMessageChunk(content=content)
    elif role == "function" or default_class == FunctionMessageChunk:
        return FunctionMessageChunk(content=content, name=_dict["name"])
    elif role or default_class == ChatMessageChunk:
        return ChatMessageChunk(content=content, role=role)
    else:
        return default_class(content=content)


class GigaMessage:
    message = ""

    def __init__(self, message: str):
        self.message = message


class GigaChat(BaseChatModel):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain.chat_models import GigaChat
            giga = GigaChat(user="username", password="password")
    """

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {"token": "GIGA_TOKEN", "user": "GIGA_USER", "password": "GIGA_PASSWORD"}

    @property
    def lc_serializable(self) -> bool:
        return True

    api_url: str = "https://beta.saluteai.sberdevices.ru"
    model: str = LATEST_MODEL
    profanity: bool = True
    temperature: float = 0
    token: str = os.environ.get("GIGA_TOKEN", "")
    user: str = os.environ.get("GIGA_USER", "")
    password: str = os.environ.get("GIGA_PASSWORD", "")
    verbose: bool = False
    timeout: int = 600
    streaming: bool = False
    """ Whether to stream the results or not. """
    max_tokens: int = 0
    """ Maximum number of tokens to generate """
    stop_on_censor: bool = True
    """ Stop generation and throw exception if censor is detected """
    censor_finish_reason: List[str] = ["request_censor", "request_blacklist"]
    """ Check certificates for all rrequests """

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # TODO: Добавить асинхронность, когда будет готов класс для запросов
        return self._generate(messages, stop, None, **kwargs)

    def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        # TODO: Добавить асинхронность, когда будет готов класс для запросов
        raise Exception('Async stream is not supported yet"')

    verify_tsl: bool = True

    logger = logging.getLogger(__name__)

    def __init__(self, *args, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(*args, **kwargs)

        if not self.verify_tsl:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    @property
    def _llm_type(self) -> str:
        return "giga-chat-model"

    @classmethod
    def transform_output(cls, response: Any, stream: bool = False) -> Tuple[str, str]:
        """Transforms API response to extract desired output."""
        choice = response["choices"][0]
        if stream:
            return choice["delta"]["content"], choice.get("finish_reason", "")
        else:
            return choice["message"]["content"], choice["finish_reason"]

    @classmethod
    def convert_message_to_dict(cls, message: BaseMessage) -> dict:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs[
                    "function_call"
                ]
                if message_dict["content"] == "":
                    message_dict.pop("content")
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {
                "role": "function",
                "content": message.content,
                "name": message.name,
            }
        else:
            raise TypeError(f"Got unknown type {message}")

        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]

        return message_dict

    def _authorize(self) -> None:
        if self.user is None or self.password is None:
            raise ValueError(
                "Please provide GIGA_USER and GIGA_PASSWORD environment variables."
            )

        response = requests.request(
            "POST",
            f"{self.api_url}/v1/token",
            auth=(self.user, self.password),
            data=[],
            timeout=self.timeout,
            verify=self.verify_tsl,
        )
        if not response.ok:
            raise ValueError(
                f"Can't authorize to GigaChat. Error code: {response.status_code}"
            )

        self.token = response.json()["tok"]
        return

    @retry(
        retry=retry_if_not_exception_type(PermissionError), stop=stop_after_attempt(3)
    )
    def get_models(self) -> List[str]:
        if not self.token:
            self._authorize()

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        response = requests.get(
            f"{self.api_url}/v1/models",
            headers=headers,
            timeout=600,
            verify=self.verify_tsl,
        )
        if not response.ok:
            if self.verbose:
                self.logger.warning(
                    "Giga error: %i %s", response.status_code, response.text
                )
            if response.status_code == 401:
                self.token = ""
            raise ValueError(
                f"Can't get response from GigaChat. Error code: {response.status_code}"
            )

        return [model["id"] for model in response.json()["data"]]

    @staticmethod
    def convert_dict_to_message(_dict: Mapping[str, Any]) -> BaseMessage:
        role = _dict["role"]
        if role == "user":
            return HumanMessage(content=_dict["content"])
        elif role == "assistant":
            # Fix for azure
            # Also OpenAI returns None for tool invocations
            content = _dict.get("content", "") or ""
            if _dict.get("function_call"):
                additional_kwargs = {"function_call": dict(_dict["function_call"])}
            else:
                additional_kwargs = {}
            return AIMessage(content=content, additional_kwargs=additional_kwargs)
        elif role == "system":
            return SystemMessage(content=_dict["content"])
        elif role == "function":
            return FunctionMessage(content=_dict["content"], name=_dict["name"])
        else:
            return ChatMessage(content=_dict["content"], role=role)

    def _create_chat_result(self, response: Mapping[str, Any]) -> ChatResult:
        generations = []
        for res in response["choices"]:
            message = self.convert_dict_to_message(res["message"])
            finish_reason = res.get("finish_reason")
            gen = ChatGeneration(
                message=message,
                generation_info=dict(finish_reason=finish_reason),
            )
            generations.append(gen)
            if finish_reason != "stop":
                self.logger.warning(
                    "Giga generation stopped with reason: %s",
                    finish_reason,
                )

            if self.verbose:
                self.logger.warning("Giga response: %s", message.content)
        token_usage = response.get("usage", {})
        llm_output = {"token_usage": token_usage, "model_name": self.model}
        return ChatResult(generations=generations, llm_output=llm_output)

    def _request(self, messages: List[BaseMessage]) -> Any:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.token}",
        }

        message_dicts = [self.convert_message_to_dict(m) for m in messages]
        payload = {
            "model": self.model,
            "profanity_check": self.profanity,
            "messages": message_dicts,
        }
        if self.temperature > 0:
            payload["temperature"] = self.temperature
        if self.max_tokens > 0:
            payload["max_tokens"] = self.max_tokens
        if self.streaming:
            payload["stream"] = True

        if self.verbose:
            self.logger.warning("Giga request: %s", payload)

        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=600,
            verify=self.verify_tsl,
            stream=self.streaming,
        )
        if not response.ok:
            if self.verbose:
                self.logger.warning(
                    "Giga error: %i %s", response.status_code, response.text
                )
            if response.status_code == 401:
                self.token = ""
            raise ValueError(
                f"Can't get response from GigaChat. Error code: {response.status_code}"
            )
        if self.streaming and "text/event-stream" in response.headers.get(
            "Content-Type", ""
        ):
            return (json.loads(line) for line in parse_stream(response.iter_lines()))
        return response.json()

    @retry(
        retry=retry_if_not_exception_type(PermissionError),
        stop=stop_after_attempt(3),
        reraise=True,
    )
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Calls the GigaChat API to get the response.
        """
        if not self.token:
            self._authorize()

        if self.model is None:
            self.model = self.get_models()[0]

        if self.streaming:
            generation: Optional[ChatGenerationChunk] = None
            for chunk in self._stream(messages, stop, run_manager, **kwargs):
                if generation is None:
                    generation = chunk
                else:
                    generation += chunk
            assert generation is not None
            return ChatResult(generations=[generation])
        else:
            response = self._request(messages)

            return self._create_chat_result(response)

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[CallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        if not self.token:
            self._authorize()

        if self.model is None:
            self.model = self.get_models()[0]

        default_chunk_class = AIMessageChunk

        for chunk in self._request(
            messages=messages,
        ):
            if len(chunk["choices"]) == 0:
                continue
            choice = chunk["choices"][0]
            chunk = _convert_delta_to_message_chunk(
                choice["delta"], default_chunk_class
            )
            finish_reason = choice.get("finish_reason")
            generation_info = (
                dict(finish_reason=finish_reason) if finish_reason is not None else None
            )
            default_chunk_class = chunk.__class__
            yield ChatGenerationChunk(message=chunk, generation_info=generation_info)
            if run_manager:
                run_manager.on_llm_new_token(
                    chunk.content, chunk=ChatGenerationChunk(message=chunk)
                )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Returns parameters identifying the model."""
        return {}

    def get_num_tokens(self, text: str) -> int:
        return round(len(text) / 4.6)
