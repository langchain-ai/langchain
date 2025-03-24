import json
from typing import Any, Iterator, List, Optional

from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)
from pydantic import PrivateAttr


class ChatSeekrFlow(BaseChatModel):
    """
    Implements the BaseChatModel interface for SeekrFlow's LLMs,
    with separate methods for single vs. streaming responses.
    """

    model_name: str
    temperature: Optional[float] = 0.7
    streaming: bool = False

    _client: Any = PrivateAttr()

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.7,
        streaming: bool = False,
        **kwargs,
    ):
        if client is None:
            raise ValueError("SeekrFlow client cannot be None.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("A valid model name must be provided.")

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            streaming=streaming,
            **kwargs,
        )
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "seekrflow"

    def invoke(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> AIMessage:
        """Return a single final AIMessage from the SeekrFlow API."""
        system_content = next(
            (msg.content for msg in messages if isinstance(msg, SystemMessage)), None
        )
        user_content = " ".join(
            msg.content for msg in messages if isinstance(msg, HumanMessage)
        )

        api_messages = []
        if system_content:
            api_messages.append({"role": "system", "content": system_content})
        api_messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            model=self.model_name, messages=api_messages, temperature=self.temperature
        )

        ai_content = response.choices[0].message.content

        if stop:
            stop_positions = [ai_content.find(token) for token in stop if token in ai_content]
            if stop_positions:
                ai_content = ai_content[:min(pos for pos in stop_positions if pos != -1)]

        return AIMessage(content=ai_content)

    def stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> Iterator[AIMessage]:
        """Yield AIMessage chunks when streaming=True, with stop token support."""
        if not self.streaming:
            raise ValueError("Streaming is disabled. Cannot call .stream().")

        system_content = next(
            (msg.content for msg in messages if isinstance(msg, SystemMessage)), None
        )
        user_content = " ".join(
            msg.content for msg in messages if isinstance(msg, HumanMessage)
        )

        api_messages = []
        if system_content:
            api_messages.append({"role": "system", "content": system_content})
        api_messages.append({"role": "user", "content": user_content})

        buffer = ""

        response_stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in response_stream:
            try:
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue

                choice = chunk.choices[0]
                if hasattr(choice, "delta") and choice.delta and hasattr(choice.delta, "content"):
                    content = choice.delta.content
                    if content:
                        buffer += content
                        if stop:
                            if any(stop_token in buffer for stop_token in stop):
                                for token in stop:
                                    idx = buffer.find(token)
                                    if idx != -1:
                                        buffer = buffer[:idx]
                                        yield AIMessage(content=buffer)
                                        return
                        yield AIMessage(content=content)
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            except Exception:
                continue

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs,
    ) -> ChatResult:
        """Implements the required generate function."""
        response_msg = self.invoke(messages, stop=stop, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=response_msg)], llm_output={}
        )

