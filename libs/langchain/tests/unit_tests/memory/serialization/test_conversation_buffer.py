import json

import pytest

from langchain.memory import ConversationBufferMemory

SERIALIZED_MEMORY_JSON = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "memory", "buffer", "ConversationBufferMemory"],
    "kwargs": {},
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
                            "content": "what up",
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
        "memory_key": "history",
        "output_key": None,
        "return_messages": False,
    },
}


@pytest.fixture()
def memory() -> ConversationBufferMemory:
    memory = ConversationBufferMemory()
    memory.save_context({"input": "hi"}, {"output": "what up"})
    memory.load_memory_variables({})
    return memory


def test_conversion_to_json(memory: ConversationBufferMemory) -> None:
    assert memory.to_json() == SERIALIZED_MEMORY_JSON


def test_conversion_from_json(memory: ConversationBufferMemory) -> None:
    json_str = json.dumps(SERIALIZED_MEMORY_JSON)
    revived_obj = ConversationBufferMemory.from_json(json_str)
    assert revived_obj == memory
