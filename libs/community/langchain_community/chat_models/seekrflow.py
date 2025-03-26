import json
from typing import Any, Iterator, List, Optional

from pydantic import Field, PrivateAttr

from langchain.chat_models.base import BaseChatModel
from langchain.schema import (
    AIMessage,
    BaseMessage,
    ChatGeneration,
    ChatResult,
    HumanMessage,
    SystemMessage,
)


class ChatSeekrFlow(BaseChatModel):
    """
    Implements the BaseChatModel interface for SeekrFlow's LLMs,
    with separate methods for single vs. streaming responses.
    """

    model_name: str = Field(default="")
    temperature: float = 0.7
    streaming: bool = False

    _client: Any = PrivateAttr()

    def __init__(
        self,
        *,
        client: Any,
        model_name: str,
        temperature: float = 0.7,
        streaming: bool = False,
        **kwargs: Any,
    ) -> None:
        if client is None:
            raise ValueError("SeekrFlow client cannot be None.")
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("A valid model name must be provided.")

        super().__init__(**kwargs)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "temperature", temperature)
        object.__setattr__(self, "streaming", streaming)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "seekrflow"

    def _convert_input(self, input: Any) -> List[BaseMessage]:
        """
        Convert various input types into a list of BaseMessage.
        Supports str, list[BaseMessage], dict with "text", or an object with to_messages().
        """
        if isinstance(input, list):
            return input
        elif isinstance(input, str):
            return [HumanMessage(content=input)]
        elif isinstance(input, dict):
            if "text" in input:
                return [HumanMessage(content=input["text"])]
            elif hasattr(input, "to_messages") and callable(input.to_messages):
                return input.to_messages()
        elif hasattr(input, "to_messages") and callable(input.to_messages):
            return input.to_messages()
        raise ValueError(
            "Invalid input type; expected str, List[BaseMessage], dict with 'text', "
            "or ChatPromptValue."
        )

    def invoke(
        self,
        input: Any,
        config: Optional[Any] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> AIMessage:
        """Return a single final AIMessage from the SeekrFlow API."""
        messages = self._convert_input(input)
        system_content = next(
            (msg.content for msg in messages if isinstance(msg, SystemMessage)),
            None,
        )
        user_content = " ".join(
            msg.content for msg in messages if isinstance(msg, HumanMessage)
        )

        api_messages = []
        if system_content:
            api_messages.append({"role": "system", "content": system_content})
        api_messages.append({"role": "user", "content": user_content})

        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
        )
        ai_content = response.choices[0].message.content
        if stop:
            stop_positions = [
                ai_content.find(token) for token in stop if token in ai_content
            ]
            if stop_positions:
                ai_content = ai_content[
                    : min(pos for pos in stop_positions if pos != -1)
                ]
        return AIMessage(content=ai_content)

    def stream(
        self,
        input: Any,
        config: Optional[Any] = None,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessage]:
        """Yield AIMessage chunks when streaming=True, with stop token support."""
        if not self.streaming:
            raise ValueError("Streaming is disabled. Cannot call .stream().")
        messages = self._convert_input(input)
        system_content = next(
            (msg.content for msg in messages if isinstance(msg, SystemMessage)),
            None,
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
                if (
                    hasattr(choice, "delta")
                    and choice.delta
                    and hasattr(choice.delta, "content")
                ):
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
        input: Any,
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Implements the required generate function."""
        response_msg = self.invoke(input, stop=stop, **kwargs)
        return ChatResult(
            generations=[ChatGeneration(message=response_msg)], llm_output={}
        )
