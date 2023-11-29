import json

import pytest

from langchain.memory import ConversationSummaryBufferMemory
from tests.unit_tests.llms.fake_llm import FakeLLM

SERIALIZED_MEMORY_JSON = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "memory", "summary_buffer", "ConversationSummaryBufferMemory"],
    "kwargs": {"llm": "FakeLLM", "max_token_limit": 10},
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
        "max_token_limit": 10,
        "memory_key": "history",
        "moving_summary_buffer": "foo",
        "output_key": None,
        "prompt": {
            "id": ["langchain_core", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["new_lines", "summary"],
                "template": (
                    "Progressively summarize the lines of conversation provided,"
                    " adding onto the previous "
                    "summary returning a new summary.\n\nEXAMPLE\nCurrent "
                    "summary:\nThe human asks what "
                    "the AI thinks of artificial intelligence. The AI thinks "
                    "artificial intelligence is "
                    "a force for good.\n\nNew lines of conversation:\nHuman: "
                    "Why do you think artificial "
                    "intelligence is a force for good?\nAI: Because artificial "
                    "intelligence will help humans "
                    "reach their full potential.\n\nNew summary:\nThe human asks"
                    " what the AI thinks of "
                    "artificial intelligence. The AI thinks artificial "
                    "intelligence is a force for good "
                    "because it will help humans reach their full potential.\nEND"
                    " OF EXAMPLE\n\nCurrent summary:"
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
def memory() -> ConversationSummaryBufferMemory:
    memory = ConversationSummaryBufferMemory(llm=FakeLLM(), max_token_limit=10)
    memory.save_context({"input": "hi"}, {"output": "what's up"})
    memory.save_context({"input": "not much you"}, {"output": "not much"})
    memory.load_memory_variables({})
    return memory


def test_conversion_to_json(memory: ConversationSummaryBufferMemory) -> None:
    assert memory.to_json() == SERIALIZED_MEMORY_JSON


def test_conversion_from_json(memory: ConversationSummaryBufferMemory) -> None:
    llm = FakeLLM()
    json_str = json.dumps(SERIALIZED_MEMORY_JSON)
    revived_obj = ConversationSummaryBufferMemory.from_json(json_str, llm=llm)
    assert revived_obj == memory
