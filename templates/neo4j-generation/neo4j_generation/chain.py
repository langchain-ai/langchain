from typing import List, Optional

from langchain_community.graphs import Neo4jGraph
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_openai import ChatOpenAI

graph = Neo4jGraph()


llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)


def chain(
    text: str,
    allowed_nodes: Optional[List[str]] = None,
    allowed_relationships: Optional[List[str]] = None,
) -> str:
    """
    Process the given text to extract graph data and constructs a graph document from the extracted information.
    The constructed graph document is then added to the graph.

    Parameters:
    - text (str): The input text from which the information will be extracted to construct the graph.
    - allowed_nodes (Optional[List[str]]): A list of node labels to guide the extraction process.
                                   If not provided, extraction won't have specific restriction on node labels.
    - allowed_relationships (Optional[List[str]]): A list of relationship types to guide the extraction process.
                                  If not provided, extraction won't have specific restriction on relationship types.

    Returns:
    str: A confirmation message indicating the completion of the graph construction.
    """  # noqa: E501
    # Construct document based on text
    documents = [Document(page_content=text)]
    # Extract graph data using OpenAI functions
    llm_graph_transformer = LLMGraphTransformer(
        llm=llm,
        allowed_nodes=allowed_nodes,
        allowed_relationships=allowed_relationships,
    )
    graph_documents = llm_graph_transformer.convert_to_graph_documents(documents)
    # Store information into a graph
    graph.add_graph_documents(graph_documents)
    return "Graph construction finished"
