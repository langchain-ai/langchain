# flake8: noqa
from langchain.memory.prompt import (
    ENTITY_EXTRACTION_PROMPT,
    ENTITY_MEMORY_CONVERSATION_TEMPLATE,
    ENTITY_SUMMARIZATION_PROMPT,
    KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT,
    SUMMARY_PROMPT,
)
from langchain_core.prompts.prompt import PromptTemplate

DEFAULT_TEMPLATE = """Ниже приводится дружеский разговор между человеком и AI. AI разговорчив и предоставляет множество конкретных деталей из своего контекста. Если AI не знает ответа на вопрос, он честно говорит, что не знает.

Текущий разговор:
{history}
Human: {input}
AI:"""
PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

# Only for backwards compatibility

__all__ = [
    "SUMMARY_PROMPT",
    "ENTITY_MEMORY_CONVERSATION_TEMPLATE",
    "ENTITY_SUMMARIZATION_PROMPT",
    "ENTITY_EXTRACTION_PROMPT",
    "KNOWLEDGE_TRIPLE_EXTRACTION_PROMPT",
    "PROMPT",
]
