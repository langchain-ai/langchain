import logging
from typing import Any, List, Mapping, cast
from uuid import uuid4

import requests
from langchain.schema import AIMessage, ChatGeneration, ChatResult, HumanMessage
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessageChunk, BaseMessage, ToolCall
from langchain_core.messages.tool import tool_call
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import Field

# Initialize logging
_logger = logging.getLogger(__name__)


def _get_tool_calls_from_response(response: Mapping[str, Any]) -> List[ToolCall]:
    """Get tool calls from ollama response."""
    tool_calls = []
    if "tool_calls" in response.json()["result"]:
        for tc in response.json()["result"]["tool_calls"]:
            tool_calls.append(
                tool_call(
                    id=str(uuid4()),
                    name=tc["name"],
                    args=tc["arguments"],
                )
            )
    return tool_calls


class CloudflareWorkersAIChatModel(BaseChatModel):
    """Custom chat model for Cloudflare Workers AI"""

    account_id: str = Field(...)
    api_token: str = Field(...)
    model: str = Field(...)

    def __init__(self, **kwargs):
        """Initialize with necessary credentials."""
        super().__init__(**kwargs)

    def _generate(
        self, messages: List[BaseMessage], stop: List[str] = None, **kwargs: Any
    ) -> ChatResult:
        """Generate a response based on the messages provided."""
        prompt = self._format_messages(messages)
        _logger.info(f"Sending prompt to Cloudflare Workers AI: {prompt}")

        headers = {"Authorization": f"Bearer {self.api_token}"}
        if "tools" in kwargs:
            tool = kwargs["tools"]
            data = {"prompt": prompt, "tools": [tool]}
        else:
            data = {"prompt": prompt}
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"

        response = requests.post(url, headers=headers, json=data)
        tool_calls = _get_tool_calls_from_response(response)
        ai_message = AIMessage(
            content=response, tool_calls=cast(AIMessageChunk, tool_calls)
        )
        chat_generation = ChatGeneration(message=ai_message)
        return ChatResult(generations=[chat_generation])

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """Helper function to format messages into a prompt."""
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append(f"Human: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"AI: {message.content}")
        return "\n".join(formatted_messages)

    def bind_tools(
        self, tools: List[BaseTool], **kwargs: Any
    ) -> Runnable[LanguageModelInput, BaseMessage]:
        """Bind tools for use in model generation."""
        formatted_tools = [convert_to_openai_tool(tool) for tool in tools]
        return super().bind(tools=formatted_tools, **kwargs)

    @property
    def _llm_type(self) -> str:
        """Return the type of the LLM (for Langchain compatibility)."""
        return "cloudflare-workers-ai"
