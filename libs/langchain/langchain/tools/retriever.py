from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.retrievers import BaseRetriever

from langchain.tools import Tool


class RetrieverInput(BaseModel):
    query: str = Field(description="query to look up in retriever")


def create_retriever_tool(
    retriever: BaseRetriever, name: str, description: str
) -> Tool:
    """Create a tool to do retrieval of documents.

    Args:
        retriever: The retriever to use for the retrieval
        name: The name for the tool. This will be passed to the language model,
            so should be unique and somewhat descriptive.
        description: The description for the tool. This will be passed to the language
            model, so should be descriptive.

    Returns:
        Tool class to pass to an agent
    """
    return Tool(
        name=name,
        description=description,
        func=retriever.get_relevant_documents,
        coroutine=retriever.aget_relevant_documents,
        args_schema=RetrieverInput,
    )
