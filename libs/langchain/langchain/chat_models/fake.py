"""Fake ChatModel for testing purposes."""
import asyncio
import time
from typing import Any
from typing import AsyncIterator
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

from langchain.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.chat_models.base import SimpleChatModel
from langchain.schema.messages import AIMessage
from langchain.schema.messages import AIMessageChunk
from langchain.schema.messages import BaseMessage
from langchain.schema.output import ChatGeneration
from langchain.schema.output import ChatGenerationChunk
from langchain.schema.output import ChatResult


class FakeListChatModel(SimpleChatModel):
    """Fake ChatModel for testing purposes."""

    responses: List
    sleep: Optional[float] = None
    i: int = -1

    def get_next_response(self):
        self.i = (self.i + 1) % len(self.responses)
        return self.responses[self.i]

    @property
    def _llm_type(self) -> str:
        return "fake-list-chat-model"

    def _call(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """First try to lookup in queries, else return 'foo' or 'bar'."""
        return self.get_next_response()

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[CallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        response = self.get_next_response()
        for c in response:
            if self.sleep is not None:
                time.sleep(self.sleep)
            yield ChatGenerationChunk(message=AIMessageChunk(content=c))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Union[List[str], None] = None,
        run_manager: Union[AsyncCallbackManagerForLLMRun, None] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        response = self.get_next_response()
        for c in response:
            if self.sleep is not None:
                await asyncio.sleep(self.sleep)
            yield ChatGenerationChunk(message=AIMessageChunk(content=c))

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "responses": self.responses
        }

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        response = self.get_next_response()
        print(response)

        if isinstance(response, dict):
            message = AIMessage(
                content="",
                additional_kwargs={
                    "function_call": response
                },
            )
        elif isinstance(response, str):
            message = AIMessage(content=response)
        elif isinstance(response, AIMessage):
            message = response
        else:
            raise ValueError(f"Invalid response type: {type(response)}")

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])
