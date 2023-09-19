# noqa: I001
"""
GigaChatModel for GigaChat.
"""

import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import requests
import urllib3.exceptions
from tenacity import retry, retry_if_not_exception_type, stop_after_attempt

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    ChatMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain.schema.output import ChatGenerationChunk

LATEST_MODEL = "GigaChat:latest"


class GigaChat(SimpleChatModel):
    """`GigaChat` large language models API.

    To use, you should pass login and password to access GigaChat API or use token.

    Example:
        .. code-block:: python

            from langchain.chat_models import GigaChat
            giga = GigaChat(user="username", password="password")
    """

    api_url: str = "https://beta.saluteai.sberdevices.ru"
    model: str = LATEST_MODEL
    profanity: bool = True
    temperature: float = 0
    token: str = os.environ.get("GIGA_TOKEN", "")
    user: str = os.environ.get("GIGA_USER", "")
    password: str = os.environ.get("GIGA_PASSWORD", "")
    verbose: bool = False
    timeout: int = 600
    max_tokens: int = 0
    """ Maximum number of tokens to generate """
    stop_on_censor: bool = True
    """ Stop generation and throw exception if censor is detected """
    censor_finish_reason: List[str] = ["request_censor", "request_blacklist"]
    """ Check certificates for all rrequests """
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
    def transform_output(cls, response: Any) -> Tuple[str, str]:
        """Transforms API response to extract desired output."""
        choice = response.json()["choices"][0]
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

    @retry(
        retry=retry_if_not_exception_type(PermissionError), stop=stop_after_attempt(3)
    )
    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """
        Calls the GigaChat API to get the response.
        """
        if not self.token:
            self._authorize()

        if self.model is None:
            self.model = self.get_models()[0]

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

        if self.verbose:
            self.logger.warning("Giga request: %s", payload)

        response = requests.post(
            f"{self.api_url}/v1/chat/completions",
            headers=headers,
            json=payload,
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
        text, finish_reason = self.transform_output(response)
        if finish_reason != "stop":
            self.logger.warning(
                "Giga generation stopped \
with reason: %s",
                finish_reason,
            )

        if self.stop_on_censor and finish_reason in self.censor_finish_reason:
            raise PermissionError("Censor detected")

        if self.verbose:
            self.logger.warning("Giga response: %s", text)

        return text

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[CallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=AIMessageChunk(content="Async stream is not supported yet")
        )

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Returns parameters identifying the model."""
        return {}

    def get_num_tokens(self, text: str) -> int:
        return round(len(text) / 4.6)
