import json

import pytest

from langchain.memory import ConversationSummaryMemory
from tests.unit_tests.llms.fake_llm import FakeLLM

SERIALIZED_MEMORY_JSON = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "memory", "summary", "ConversationSummaryMemory"],
    "kwargs": {"llm": "FakeLLM"},
    "obj": {
        "ai_prefix": "AI",
        "buffer": "foo",
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
                            "content": "what is up",
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
        "llm": {
            "id": ["tests", "unit_tests", "llms", "fake_llm", "FakeLLM"],
            "lc": 1,
            "repr": "FakeLLM()",
            "type": "not_implemented",
        },
        "memory_key": "history",
        "output_key": None,
        "prompt": {
            "id": ["langchain_core", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["new_lines", "summary"],
                "template": (
                    "Progressively summarize the lines of conversation provided, "
                    "adding onto the previous "
                    "summary returning a new summary.\n\nEXAMPLE\nCurrent "
                    "summary:\nThe human asks what "
                    "the AI thinks of artificial intelligence. The AI thinks "
                    "artificial intelligence is "
                    "a force for good.\n\nNew lines of conversation:\nHuman: "
                    "Why do you think artificial "
                    "intelligence is a force for good?\nAI: Because artificial "
                    "intelligence will help humans "
                    "reach their full potential.\n\nNew summary:\nThe human asks "
                    "what the AI thinks of "
                    "artificial intelligence. The AI thinks artificial intelligence"
                    " is a force for good "
                    "because it will help humans reach their full potential.\nEND "
                    "OF EXAMPLE\n\nCurrent summary:"
                    "\n{summary}\n\nNew lines of conversation:\n{new_lines}\n\nNew"
                    " summary:"
                ),
                "template_format": "f-string",
            },
            "lc": 1,
            "type": "constructor",
        },
        "return_messages": False,
        "summary_message_cls": "SystemMessage",
    },
}


@pytest.fixture()
def memory() -> ConversationSummaryMemory:
    memory = ConversationSummaryMemory(llm=FakeLLM())
    memory.save_context({"input": "hi"}, {"output": "what is up"})
    memory.load_memory_variables({})
    return memory


def test_conversion_to_json(memory: ConversationSummaryMemory) -> None:
    assert memory.to_json() == SERIALIZED_MEMORY_JSON


def test_conversion_from_json(memory: ConversationSummaryMemory) -> None:
    llm = FakeLLM()
    json_str = json.dumps(SERIALIZED_MEMORY_JSON)
    revived_obj = ConversationSummaryMemory.from_json(json_str, llm=llm)
    assert revived_obj == memory
