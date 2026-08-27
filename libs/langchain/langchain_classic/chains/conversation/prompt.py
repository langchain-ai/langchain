from langchain_core.prompts.prompt import PromptTemplate

from langchain_classic.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    ENTITY_SUMMARIZATION_PROMPT,
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
    SUMMARY_PROMPT,
)

DEFAULT_TEMPLATE = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:"""  # noqa: E501
PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

# Only for backwards compatibility

__all__ = [
    "ENTITY_EXTRACTION_PROMPT",
    "ENTITY_MEMORY_CONVERSATION_TEMPLATE",
    "ENTITY_SUMMARIZATION_PROMPT",
    "KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT",
    "PROMPT",
    "SUMMARY_PROMPT",
]
