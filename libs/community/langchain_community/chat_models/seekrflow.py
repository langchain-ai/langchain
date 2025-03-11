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

    # Pydantic fields
    model_name: str  # Required
    temperature: Optional[float] = 0.7  # Default to 0.7
    streaming: bool = False  # Default to False

    # Private attribute for the SeekrFlow client
    _client: Any = PrivateAttr()

    def __init__(
        self,
        client: Any,
        model_name: str,
        temperature: float = 0.7,
        streaming: bool = False,
        **kwargs,
    ):
        """
        Initialize the SeekrFlow chat model.

        Args:
            client (Any): SeekrFlow API client.
            model_name (str): The model name to use (required).
            temperature (float, optional): Sampling temperature (default: 0.7).
            streaming (bool, optional): Enable streaming (default: False).

        Raises:
            ValueError: If `client` is None or `model_name` is empty.
        """
        # Ensure the client is provided
        if client is None:
            raise ValueError("SeekrFlow client cannot be None.")

        # Ensure model_name is valid
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("A valid model name must be provided.")

        # Initialize the parent Pydantic model
        super().__init__(
            model_name=model_name,
            temperature=temperature,
            streaming=streaming,
            **kwargs,
        )

        # Store the client separately (since it's not a Pydantic field)
        self._client = client

    @property
    def _llm_type(self) -> str:
        return "seekrflow"

    #
    # Single-call synchronous logic
    #
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
            for token in stop:
                idx = ai_content.find(token)
                if idx != -1:
                    ai_content = ai_content[:idx]
                    break

        return AIMessage(content=ai_content)

    #
    # Streaming logic
    #
    def stream(
        self, messages: List[BaseMessage], stop: Optional[List[str]] = None, **kwargs
    ) -> Iterator[AIMessage]:
        """Yield AIMessage chunks when streaming=True."""
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

        response_stream = self._client.chat.completions.create(
            model=self.model_name,
            messages=api_messages,
            temperature=self.temperature,
            stream=True,
        )

        for chunk in response_stream:
            try:
                # ✅ Ensure chunk is an instance of expected response type
                if not hasattr(chunk, "choices") or not chunk.choices:
                    continue  # Skip empty or malformed chunks

                choice = chunk.choices[0]
                if (
                    hasattr(choice, "delta")
                    and choice.delta
                    and hasattr(choice.delta, "content")
                ):
                    content = choice.delta.content
                    if content:
                        yield AIMessage(content=content)

            except (json.JSONDecodeError, UnicodeDecodeError):
                # ✅ Handle known JSON decoding issues without breaking the stream
                continue  # Skip malformed chunks and keep streaming

            except Exception:
                # ✅ Catch all unexpected errors without stopping the entire stream
                continue  # Skip the problematic chunk but keep yielding

    #
    # Required by BaseChatModel
    #
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        config: Optional[dict] = None,
    ) -> ChatResult:
        """Implements the required generate function."""
        config = config if isinstance(config, dict) else {}
        response_msg = self.invoke(messages, stop=stop, **config)

        return ChatResult(
            generations=[ChatGeneration(message=response_msg)], llm_output={}
        )
