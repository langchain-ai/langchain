"""``ChatParrotLinkV1`` implementation for standard-tests with v1 messages.

This module provides a test implementation of ``BaseChatModel`` that supports the new
v1 message format with content blocks.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Optional, cast

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.callbacks.manager import AsyncCallbackManagerForLLMRun
from langchain_core.messages.ai import UsageMetadata
from langchain_core.v1.chat_models import BaseChatModel
from langchain_core.v1.messages import AIMessage, AIMessageChunk, MessageV1
from pydantic import Field


class ChatParrotLinkV1(BaseChatModel):
    """A custom v1 chat model that echoes input with content blocks support.

    This model is designed for testing the v1 message format and content blocks. Echoes
    the first ``parrot_buffer_length`` characters of the input and returns them as
    proper v1 content blocks.

    Example:
        .. code-block:: python

            model = ChatParrotLinkV1(parrot_buffer_length=10, model="parrot-v1")
            result = model.invoke([HumanMessage(content="hello world")])
            # Returns AIMessage with content blocks format
    """

    model_name: str = Field(alias="model")
    """The name of the model."""
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    timeout: Optional[int] = None
    stop: Optional[list[str]] = None
    max_retries: int = 2

    parrot_buffer_length: int = Field(default=50)
    """The number of characters from the last message to echo."""

    def _invoke(
        self,
        messages: list[MessageV1],
        **kwargs: Any,
    ) -> AIMessage:
        """Generate a response by echoing the input as content blocks.

        Args:
            messages: List of v1 messages to process.
            **kwargs: Additional generation parameters.

        Returns:
            AIMessage with content blocks format.
        """
        _ = kwargs  # Mark as used

        if not messages:
            return AIMessage("No input provided")

        last_message = messages[-1]

        # Extract text content from the message
        text_content = ""
        for block in last_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_content += str(block.get("text", ""))

        # Echo the first parrot_buffer_length characters
        echoed_text = text_content[: self.parrot_buffer_length]

        # Calculate usage metadata
        total_input_chars = sum(
            len(str(msg.content))
            if isinstance(msg.content, str)
            else (
                sum(len(str(block)) for block in msg.content)
                if isinstance(msg.content, list)
                else 0
            )
            for msg in messages
        )

        usage_metadata = UsageMetadata(
            input_tokens=total_input_chars,
            output_tokens=len(echoed_text),
            total_tokens=total_input_chars + len(echoed_text),
        )

        return AIMessage(
            content=echoed_text,
            response_metadata=cast(
                Any,
                {
                    "model_name": self.model_name,
                    "time_in_seconds": 0.1,
                },
            ),
            usage_metadata=usage_metadata,
        )

    def _stream(
        self,
        messages: list[MessageV1],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[AIMessageChunk]:
        """Stream the response by yielding character chunks.

        Args:
            messages: List of v1 messages to process.
            stop: Stop sequences (unused in this implementation).
            run_manager: Callback manager for the LLM run.
            **kwargs: Additional generation parameters.

        Yields:
            AIMessageChunk objects with individual characters.
        """
        _ = stop  # Mark as used
        _ = kwargs  # Mark as used

        if not messages:
            yield AIMessageChunk("No input provided")
            return

        last_message = messages[-1]

        # Extract text content from the message
        text_content = ""
        # Extract text from content blocks
        for block in last_message.content:
            if isinstance(block, dict) and block.get("type") == "text":
                text_content += str(block.get("text", ""))

        # Echo the first parrot_buffer_length characters
        echoed_text = text_content[: self.parrot_buffer_length]

        # Calculate total input for usage metadata
        total_input_chars = sum(
            len(str(msg.content))
            if isinstance(msg.content, str)
            else (
                sum(len(str(block)) for block in msg.content)
                if isinstance(msg.content, list)
                else 0
            )
            for msg in messages
        )

        # Stream each character as a chunk
        for i, char in enumerate(echoed_text):
            usage_metadata = UsageMetadata(
                input_tokens=total_input_chars if i == 0 else 0,
                output_tokens=1,
                total_tokens=total_input_chars + 1 if i == 0 else 1,
            )

            chunk = AIMessageChunk(
                content=char,
                usage_metadata=usage_metadata,
            )

            if run_manager:
                run_manager.on_llm_new_token(char, chunk=chunk)

            yield chunk

        # Final chunk with response metadata
        final_chunk = AIMessageChunk(
            content=[],
            response_metadata=cast(
                Any,
                {
                    "model_name": self.model_name,
                    "time_in_seconds": 0.1,
                },
            ),
        )
        yield final_chunk

    async def _ainvoke(
        self,
        messages: list[MessageV1],
        **kwargs: Any,
    ) -> AIMessage:
        """Async generate a response (delegates to sync implementation).

        Args:
            messages: List of v1 messages to process.
            **kwargs: Additional generation parameters.

        Returns:
            AIMessage with content blocks format.
        """
        # For simplicity, delegate to sync implementation
        return self._invoke(messages, **kwargs)

    async def _astream(
        self,
        messages: list[MessageV1],
        stop: Optional[list[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[AIMessageChunk]:
        """Async stream the response (delegates to sync implementation).

        Args:
            messages: List of v1 messages to process.
            stop: Stop sequences (unused in this implementation).
            run_manager: Async callback manager for the LLM run.
            **kwargs: Additional generation parameters.

        Yields:
            AIMessageChunk objects with individual characters.
        """
        # For simplicity, delegate to sync implementation
        for chunk in self._stream(messages, stop, None, **kwargs):
            yield chunk

    @property
    def _llm_type(self) -> str:
        """Get the type of language model used by this chat model."""
        return "parrot-chat-model-v1"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
            "parrot_buffer_length": self.parrot_buffer_length,
        }

    def get_token_ids(self, text: str) -> list[int]:
        """Convert text to token IDs using simple character-based tokenization.

        For testing purposes, we use a simple approach where each character
        maps to its ASCII/Unicode code point.

        Args:
            text: The text to tokenize.

        Returns:
            List of token IDs (character code points).
        """
        return [ord(char) for char in text]

    def get_num_tokens(self, text: str) -> int:
        """Get the number of tokens in the text.

        Args:
            text: The text to count tokens for.

        Returns:
            Number of tokens (characters in this simple implementation).
        """
        return len(text)
