from langchain.schema.runnable import ConfigurableField

from .chain import chain
from .retriever_agent import executor

final_chain = chain.configurable_alternatives(
    ConfigurableField(id="chain"),
    default_key="response",
    # This adds a new option, with name `openai` that is equal to `ChatOpenAI()`
    retrieve=executor,
)
