from langchain.schema import BaseRetriever
from langchain.tools import Tool


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
    )
