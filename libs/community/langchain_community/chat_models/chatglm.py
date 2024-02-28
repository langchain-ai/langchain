from __future__ import annotations

from typing import Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    FunctionMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult

from langchain_community.llms.chatglm import ChatGLM


class ChatGLM_ChatModel(BaseChatModel, ChatGLM):

    """ChatGLM LLM ChatModel.
    Example:
        .. code-block:: python

            from langchain_community.chat_models.chatglm import ChatGLM_ChatModel
            llm = ChatGLM_ChatModel(temperature=0.9, top_p=0.9)
    """

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "ChatGLM"

    def _convert_messages(self, messages: List[BaseMessage]) -> str:
        if len(messages) == 0:
            raise ValueError(f"Got no message in {messages}")
        elif len(messages) == 1:
            history = []
            prompt = messages[0].content
        else:
            history = []
            prompt = messages[-1].content            
            for index, message in enumerate(messages):
                if index == len(messages) - 1:
                    break
                if isinstance(message, SystemMessage):
                    history.append([message.content, "知道了。"])
                elif isinstance(message, HumanMessage):
                    history.append([message.content, ""])
                elif isinstance(message, AIMessage):
                    history.append(["", message.content])
                elif isinstance(message, FunctionMessage):
                    history.append([message.content, "知道了。"])
                else:
                    raise ValueError(f"Got unknown type {message}")
        return history, prompt

    def _generate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> ChatResult:
        history, prompt = self._convert_messages(messages)
        response = AIMessage(content=self._call(history=history, prompt=prompt))
        return ChatResult(generations=[ChatGeneration(message=response)])
