from __future__ import annotations

from typing import Any, List, Tuple, Optional

import requests

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.pydantic_v1 import BaseModel

from langchain_community.llms.utils import enforce_stop_tokens


class ChatGLM_ChatModel(BaseChatModel, BaseModel):

    """ChatGLM LLM ChatModel.
    Example:
        .. code-block:: python

            from langchain_community.chat_models.chatglm import ChatGLM_ChatModel
            llm = ChatGLM_ChatModel(temperature=0.9, top_p=0.9)
    """
    
    streaming: bool = False
    """Whether to stream the results."""
    endpoint_url: str = "http://127.0.0.1:8000"
    """Endpoint URL to use."""
    max_token: int = 20000
    """Max token allowed to pass to the model."""
    temperature: float = 0.1
    """LLM model temperature from 0 to 10."""
    history: List[List] = []
    """History of the conversation"""
    top_p: float = 0.7
    """Top P for nucleus sampling from 0 to 1"""
    with_history: bool = False
    """Whether to use history or not"""

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "ChatGLM"

    def _convert_messages(self, messages: List[BaseMessage]) -> Tuple[List, str]:
        """Convert the list of messages into a history and a prompt to feed the ChatGLM."""
        if len(messages) == 0:
            raise ValueError(f"Got no message in {messages}")
        elif len(messages) == 1:
            self.history = []
            prompt: str = str(messages[0].content)
        else:
            self.history = []
            prompt = str(messages[-1].content)
            for index, message in enumerate(messages):
                if index == len(messages) - 1:
                    break
                if isinstance(message, SystemMessage):
                    self.history.append([message.content, "知道了。"])
                elif isinstance(message, HumanMessage):
                    self.history.append([message.content, ""])
                elif isinstance(message, AIMessage):
                    self.history.append(["", message.content])
                elif isinstance(message, FunctionMessage):
                    self.history.append([message.content, "知道了。"])
                else:
                    raise ValueError(f"Got unknown type {message}")
        return self.history, prompt

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Call the ChatGLM to generalize the response."""

        headers = {"Content-Type": "application/json"}
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "history": self.history,
            "max_length": self.max_token,
            "top_p": self.top_p,
            "endpoint_url": self.endpoint_url,
        }

        try:
            response = requests.post(self.endpoint_url, headers=headers, json=payload)
        except requests.exceptions.RequestException as e:
            raise ValueError(f"Error raised by inference endpoint: {e}")

        if response.status_code != 200:
            raise ValueError(f"Failed with response: {response}")

        try:
            parsed_response = response.json()

            # Check if response content does exists
            if isinstance(parsed_response, dict):
                content_keys = "response"
                if content_keys in parsed_response:
                    text = parsed_response[content_keys]
                else:
                    raise ValueError(f"No content in response : {parsed_response}")
            else:
                raise ValueError(f"Unexpected response type: {parsed_response}")

        except requests.exceptions.JSONDecodeError as e:
            raise ValueError(
                f"Error raised during decoding response from inference endpoint: {e}."
                f"\nResponse: {response.text}"
            )

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        if self.with_history:
            self.history = parsed_response["history"]
        else:
            self.history = []

        return text

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        history, prompt = self._convert_messages(messages)
        response = AIMessage(content=self._call(history=history, prompt=prompt))
        return ChatResult(generations=[ChatGeneration(message=response)])

