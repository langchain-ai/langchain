"""Test Xata chat memory store functionality.

Before running this test, please create a Xata database.
"""

import json
import os

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import XataChatMessageHistory
from langchain.schema.messages import _message_to_dict


class TestXata:
    @classmethod
    def setup_class(cls) -> None:
        assert os.getenv("XATA_API_KEY"), "XATA_API_KEY environment variable is not set"
        assert os.getenv("XATA_DB_URL"), "XATA_DB_URL environment variable is not set"

    def test_xata_chat_memory(self) -> None:
        message_history = XataChatMessageHistory(
            api_key=os.getenv("XATA_API_KEY", ""),
            db_url=os.getenv("XATA_DB_URL", ""),
            session_id="integration-test-session",
        )
        memory = ConversationBufferMemory(
            memory_key="baz", chat_memory=message_history, return_messages=True
        )
        # add some messages
        memory.chat_memory.add_ai_message("This is me, the AI")
        memory.chat_memory.add_user_message("This is me, the human")

        # get the message history from the memory store and turn it into a json
        messages = memory.chat_memory.messages
        messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

        assert "This is me, the AI" in messages_json
        assert "This is me, the human" in messages_json

        # remove the record from Redis, so the next test run won't pick it up
        memory.chat_memory.clear()
