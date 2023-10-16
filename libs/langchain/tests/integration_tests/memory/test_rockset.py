"""Tests RocksetChatMessageHistory by creating a collection
for message history, adding to it, and clearing it.

To run these tests, make sure you have the ROCKSET_API_KEY
and ROCKSET_REGION environment variables set.
"""

import json
import os

from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import RocksetChatMessageHistory
from langchain.schema.messages import _message_to_dict

collection_name = "langchain_demo"
session_id = "MySession"


class TestRockset:
    memory: RocksetChatMessageHistory

    @classmethod
    def setup_class(cls) -> None:
        from rockset import DevRegions, Regions, RocksetClient

        assert os.environ.get("ROCKSET_API_KEY") is not None
        assert os.environ.get("ROCKSET_REGION") is not None

        api_key = os.environ.get("ROCKSET_API_KEY")
        region = os.environ.get("ROCKSET_REGION")
        if region == "use1a1":
            host = Regions.use1a1
        elif region == "usw2a1" or not region:
            host = Regions.usw2a1
        elif region == "euc1a1":
            host = Regions.euc1a1
        elif region == "dev":
            host = DevRegions.usw2a1
        else:
            host = region

        client = RocksetClient(host, api_key)
        cls.memory = RocksetChatMessageHistory(
            session_id, client, collection_name, sync=True
        )

    def test_memory_with_message_store(self) -> None:
        memory = ConversationBufferMemory(
            memory_key="messages", chat_memory=self.memory, return_messages=True
        )

        memory.chat_memory.add_ai_message("This is me, the AI")
        memory.chat_memory.add_user_message("This is me, the human")

        messages = memory.chat_memory.messages
        messages_json = json.dumps([_message_to_dict(msg) for msg in messages])

        assert "This is me, the AI" in messages_json
        assert "This is me, the human" in messages_json

        memory.chat_memory.clear()

        assert memory.chat_memory.messages == []
