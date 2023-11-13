from langchain.pydantic_v1 import BaseModel
from langchain.schema.runnable import ConfigurableField, RunnablePassthrough

from research_assistant.search.tavily import chain as search_tavily
from research_assistant.search.web import chain as search_ddg
from research_assistant.writer import chain as writer_chain

search_chain = search_ddg.configurable_alternatives(
    ConfigurableField(id="chain"),
    default_key="duckduckgo",
    tavily=search_tavily,
)

chain_notypes = (
    RunnablePassthrough().assign(research_summary=search_chain) | writer_chain
)


class InputType(BaseModel):
    question: str


chain = chain_notypes.with_types(input_type=InputType)
