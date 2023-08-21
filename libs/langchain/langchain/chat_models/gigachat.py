"""
GigaChatModel for GigaChat.
"""

import logging
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

import requests
from langchain.callbacks.manager import (AsyncCallbackManagerForLLMRun,
                                         CallbackManagerForLLMRun)
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import (AIMessage, AIMessageChunk, BaseMessage,
                                       ChatMessage, FunctionMessage,
                                       HumanMessage, SystemMessage)
from langchain.schema.output import ChatGenerationChunk


class GigaChat(SimpleChatModel):
    api_url: str = "https://beta.saluteai.sberdevices.ru"
    model: str = "GigaChat:v1.13.0"
    profanity: bool = True
    temperature: float = 0
    token: str = os.environ.get("GIGA_TOKEN", None)
    user: str = os.environ.get("GIGA_USER", None)
    password: str = os.environ.get("GIGA_PASSWORD", None)
    verbose: bool = False

    logger = logging.getLogger(__name__)

    @property
    def _llm_type(self) -> str:
        return "giga-chat-model"

    @classmethod
    def transform_output(cls, response: Any) -> str:
        """Transforms API response to extract desired output."""
        return response.json()["choices"][0]['message']['content']

    @classmethod
    def convert_message_to_dict(cls, message: BaseMessage) -> dict:
        if isinstance(message, ChatMessage):
            message_dict = {"role": message.role, "content": message.content}
        elif isinstance(message, HumanMessage):
            message_dict = {"role": "user", "content": message.content}
        elif isinstance(message, AIMessage):
            message_dict = {"role": "assistant", "content": message.content}
            if "function_call" in message.additional_kwargs:
                message_dict["function_call"] = message.additional_kwargs["function_call"]
                if message_dict["content"] == "":
                    message_dict["content"] = None
        elif isinstance(message, SystemMessage):
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, FunctionMessage):
            message_dict = {"role": "function",
                            "content": message.content, "name": message.name}
        else:
            raise TypeError(f"Got unknown type {message}")

        if "name" in message.additional_kwargs:
            message_dict["name"] = message.additional_kwargs["name"]

        return message_dict

    def _authorize(self):
        if self.user is None or self.password is None:
            raise ValueError(
                "Please provide GIGA_USER and GIGA_PASSWORD environment variables.")

        response = requests.request(
            "POST", f"{self.api_url}/v1/token", auth=(self.user, self.password), data=[], timeout=3)
        if not response.ok:
            raise ValueError(
                f"Can't authorize to GigaChat. Error code: {response.status_code}")

        self.token = response.json()["tok"]

    def _call(self, messages: List[BaseMessage], stop: Optional[List[str]] = None, run_manager: Optional[CallbackManagerForLLMRun] = None, **kwargs: Any) -> str:
        """
        Calls the GigaChat API to get the response.
        """
        if not self.token:
            self._authorize()

        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.token}'
        }

        message_dicts = [self.convert_message_to_dict(m) for m in messages]
        payload = {
            "model": self.model,
            "profanity_check": self.profanity,
            "messages": message_dicts
        }

        if self.verbose:
            self.logger.warning(f"Giga request: {payload}")

        response = requests.post(
            f"{self.api_url}/v1/chat/completions", headers=headers, json=payload, timeout=600)
        text = self.transform_output(response)

        if self.verbose:
            self.logger.warning(f"Giga response: {text}")

        return text

    def _stream(self, messages: List[BaseMessage], stop: Union[List[str], None] = None, run_manager: Union[CallbackManagerForLLMRun, None] = None, **kwargs: Any) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="Async is not supported yet"))

    async def _astream(self, messages: List[BaseMessage], stop: Union[List[str], None] = None, run_manager: Union[AsyncCallbackManagerForLLMRun, None] = None, **kwargs: Any) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(message=AIMessageChunk(content="Async is not supported yet"))

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Returns parameters identifying the model."""
        return {}
