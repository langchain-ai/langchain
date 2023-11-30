import json

import pytest

from langchain.memory import ConversationEntityMemory
from tests.unit_tests.llms.fake_llm import FakeLLM

SERIALIZED_MEMORY_JSON = {
    "lc": 1,
    "type": "constructor",
    "id": ["langchain", "memory", "entity", "ConversationEntityMemory"],
    "kwargs": {"llm": "FakeLLM", "max_token_limit": 100},
    "obj": {
        "ai_prefix": "AI",
        "chat_history_key": "history",
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
                            "content": "whats up",
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
        "entity_cache": [],
        "entity_extraction_prompt": {
            "id": ["langchain_core", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["history", "input"],
                "template": (
                    "You are an AI assistant reading the transcript of a "
                    "conversation between an AI and a human. Extract all "
                    "of the "
                    "proper nouns from the last line of conversation. As "
                    "a guideline, "
                    "a proper noun is generally capitalized. You should "
                    "definitely "
                    "extract all names and places.\n\nThe conversation "
                    "history is "
                    'provided just in case of a coreference (e.g. "What '
                    "do you know "
                    'about him" where "him" is defined in a previous '
                    "line) -- ignore "
                    "items mentioned there that are not in the last "
                    "line.\n\nReturn "
                    "the output as a single comma-separated list, "
                    "or NONE if there is "
                    "nothing of note to return (e.g. the user is just"
                    " issuing a "
                    "greeting or having a simple conversation).\n\n"
                    "EXAMPLE\n"
                    "Conversation history:\nPerson #1: how's it going"
                    " today?\nAI: "
                    '"It\'s going great! How about you?"\nPerson #1: '
                    "good! busy "
                    'working on Langchain. lots to do.\nAI: "That sounds'
                    " like a lot "
                    "of work! What kind of things are you doing to make"
                    " Langchain "
                    "better?\"\nLast line:\nPerson #1: i'm trying to "
                    "improve Langchain's "
                    "interfaces, the UX, its integrations with various "
                    "products the "
                    "user might want ... a lot of stuff.\nOutput: "
                    "Langchain\nEND OF "
                    "EXAMPLE\n\nEXAMPLE\nConversation history:\nPerson #1: "
                    "how's it "
                    'going today?\nAI: "It\'s going great! How about you?"\n'
                    "Person #1: "
                    "good! busy working on Langchain. lots to do.\nAI: "
                    '"That sounds '
                    "like a lot of work! What kind of things are you "
                    "doing to make "
                    "Langchain better?\"\nLast line:\nPerson #1: i'm "
                    "trying to improve "
                    "Langchain's interfaces, the UX, its integrations "
                    "with various "
                    "products the user might want ... a lot of stuff. "
                    "I'm working with "
                    "Person #2.\nOutput: Langchain, Person #2\nEND OF "
                    "EXAMPLE\n\n"
                    "Conversation history (for reference only)"
                    ":\n{history}\nLast "
                    "line of conversation (for extraction):"
                    "\nHuman: {input}\n\nOutput:"
                ),
                "template_format": "f-string",
            },
            "lc": 1,
            "type": "constructor",
        },
        "entity_store": {"store": {}},
        "entity_summarization_prompt": {
            "id": ["langchain_core", "prompts", "prompt", "PromptTemplate"],
            "kwargs": {
                "input_variables": ["entity", "history", "input", "summary"],
                "template": (
                    "You are an AI assistant helping a human keep track"
                    " of facts about "
                    "relevant people, places, and concepts in their"
                    " life. Update the "
                    'summary of the provided entity in the "Entity" '
                    "section based on the "
                    "last line of your conversation with the human. "
                    "If you are writing the "
                    "summary for the first time, return a single "
                    "sentence.\nThe update "
                    "should only include facts that are relayed in"
                    " the last line of "
                    "conversation about the provided entity, and "
                    "should only contain "
                    "facts about the provided entity.\n\nIf there "
                    "is no new information "
                    "about the provided entity or the information "
                    "is not worth noting (not "
                    "an important or relevant fact to remember "
                    "long-term), return the "
                    "existing summary unchanged.\n\nFull conversation "
                    "history (for context):"
                    "\n{history}\n\nEntity to summarize:\n{entity}\n\n"
                    "Existing summary of "
                    "{entity}:\n{summary}\n\nLast line of conversation"
                    ":\nHuman: {input}\n"
                    "Updated summary:"
                ),
                "template_format": "f-string",
            },
            "lc": 1,
            "type": "constructor",
        },
        "human_prefix": "Human",
        "input_key": None,
        "k": 3,
        "llm": {
            "id": ["tests", "unit_tests", "llms", "fake_llm", "FakeLLM"],
            "lc": 1,
            "repr": "FakeLLM()",
            "type": "not_implemented",
        },
        "output_key": None,
        "return_messages": False,
    },
}


@pytest.fixture()
def memory() -> ConversationEntityMemory:
    memory = ConversationEntityMemory(llm=FakeLLM(), max_token_limit=100)
    memory.save_context({"input": "hi"}, {"output": "whats up"})
    return memory


def test_conversion_to_json(memory: ConversationEntityMemory) -> None:
    print(memory.to_json())
    assert memory.to_json() == SERIALIZED_MEMORY_JSON


def test_conversion_from_json(memory: ConversationEntityMemory) -> None:
    llm = FakeLLM()
    json_str = json.dumps(SERIALIZED_MEMORY_JSON)
    revived_obj = ConversationEntityMemory.from_json(json_str, llm=llm)
    assert revived_obj == memory
