from research_assistant.search.web import chain as search_chain
from research_assistant.writer import chain as writer_chain

chain = {
    "question": lambda x: x,
    "research_summary": search_chain
}| writer_chain
