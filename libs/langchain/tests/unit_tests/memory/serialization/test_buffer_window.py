import json

import pytest

from langchain.memory import ConversationBufferWindowMemory

SERIALIZED_MEMORY_JSON = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "memory", "buffer_window", "ConversationBufferWindowMemory"],
    "kwargs": {"k": 1},
    "obj": {
        "ai_prefix": "AI",
        "chat_memory": {
            "id": [
                "langchain",
                "memory",
                "chat_message_histories",
                "in_memory",
                "ChatMessageHistory",
            ],
            "kwargs": {},
            "lc": 1,
            "obj": {
                "messages": [
                    {
                        "data": {
                            "additional_kwargs": {},
                            "content": "hi",
                            "example": False,
                            "type": "human",
                        },
                        "type": "human",
                    },
                    {
                        "data": {
                            "additional_kwargs": {},
                            "content": "what's up",
                            "example": False,
                            "type": "ai",
                        },
                        "type": "ai",
                    },
                    {
                        "data": {
                            "additional_kwargs": {},
                            "content": "not much you",
                            "example": False,
                            "type": "human",
                        },
                        "type": "human",
                    },
                    {
                        "data": {
                            "additional_kwargs": {},
                            "content": "not much",
                            "example": False,
                            "type": "ai",
                        },
                        "type": "ai",
                    },
                ]
            },
            "repr": "",
            "type": "constructor",
        },
        "human_prefix": "Human",
        "input_key": None,
        "k": 1,
        "memory_key": "history",
        "output_key": None,
        "return_messages": False,
    },
}


@pytest.fixture()
def memory() -> ConversationBufferWindowMemory:
    memory = ConversationBufferWindowMemory(k=1)
    memory.save_context({"input": "hi"}, {"output": "what's up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})
    memory.load_memory_variables({})
    return memory


def test_to_json(memory: ConversationBufferWindowMemory) -> None:
    assert memory.to_json() == SERIALIZED_MEMORY_JSON


def test_conversion_from_json(memory: ConversationBufferWindowMemory) -> None:
    json_str = json.dumps(SERIALIZED_MEMORY_JSON)
    revived_obj = ConversationBufferWindowMemory.from_json(json_str)
    assert revived_obj == memory
