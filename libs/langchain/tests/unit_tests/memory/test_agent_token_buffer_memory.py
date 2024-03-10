from typing import Any, List

import pytest

from langchain.agents.openai_functions_agent.agent_token_buffer_memory import (
    AgentTokenBufferMemory,
)
from langchain.chat_models.base import BaseChatModel
from langchain.chat_models.fake import FakeMessagesListChatModel
from langchain.memory.chat_message_histories import ChatMessageHistory
from langchain.schema.messages import AIMessage, BaseMessage, get_buffer_string


# NOTE;tch: consider factoring it out to separate class if useful outside of this test
class FakeMessagesListChatModelWithCharCounting(FakeMessagesListChatModel):
    """Modifies token counting to count characters of text for easier testing"""

    def __init__(self, responses: List[BaseMessage], **kwargs: Any) -> None:
        super().__init__(responses=responses, **kwargs)

    def get_token_ids(self, text: str) -> List[int]:
        """Return the ordered ids of the characters in text.
        This is to simplify token counting for testing.

        Args:
            text: The string input to fake-tokenize.

        Returns:
            A list of ids corresponding to the characters in the text,
            in order they occur in the text.
        """
        return list(range(len(text)))

    def get_num_tokens_from_messages(self, messages: List[BaseMessage]) -> int:
        """Get the number of characters in the messages.
        This is to simplify token counting.

        Args:
            messages: The message inputs to tokenize.

        Returns:
            The sum of the number of characters across the messages.
        """
        # count characters, remove and compensate for prefixes
        return sum(
            [
                self.get_num_tokens(
                    get_buffer_string([m], ai_prefix="", human_prefix="")
                )
                - 2  # compensate for ': ' prefix added to the message
                for m in messages
            ]
        )


@pytest.fixture
def chat_history() -> ChatMessageHistory:
    return ChatMessageHistory(
        messages=[
            AIMessage(content="h1"),
            AIMessage(content="h2"),
            AIMessage(content="h3"),
        ]
    )


@pytest.fixture
def llm() -> BaseChatModel:
    return FakeMessagesListChatModelWithCharCounting(responses=[])


def test_memory_after_load_is_trimmed(
    chat_history: ChatMessageHistory, llm: BaseChatModel
) -> None:
    memory = AgentTokenBufferMemory(
        chat_memory=chat_history,
        max_token_limit=4,
        llm=llm,
        human_prefix="",
        ai_prefix="",
    )

    # expect only two messages,2 characters/tokens each
    assert len(memory.buffer) == 2
    # expect only last two messages from history loaded
    assert memory.buffer[0].content == "h2"
    assert memory.buffer[1].content == "h3"


def test_memory_is_kept_trimmed(
    chat_history: ChatMessageHistory, llm: BaseChatModel
) -> None:
    memory = AgentTokenBufferMemory(
        chat_memory=chat_history,
        max_token_limit=4,
        llm=llm,
        human_prefix="",
        ai_prefix="",
    )
    memory.save_context(
        inputs={"input": "i1"}, outputs={"output": "o1", "intermediate_steps": []}
    )

    memory.save_context(
        inputs={"input": "i2"}, outputs={"output": "o2", "intermediate_steps": []}
    )

    # expect only two messages,2 characters/tokens each
    assert len(memory.buffer) == 2
    # expect only last two messages
    assert memory.buffer[0].content == "i2"
    assert memory.buffer[1].content == "o2"


def test_memory_clear(chat_history: ChatMessageHistory, llm: BaseChatModel) -> None:
    memory = AgentTokenBufferMemory(
        chat_memory=chat_history,
        max_token_limit=4,
        llm=llm,
    )

    assert len(memory.buffer) > 0

    memory.clear()
    # expect only two messages,2 characters/tokens each
    assert len(memory.buffer) == 0
