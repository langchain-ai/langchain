from typing import Dict

from langchain.chains.openai_functions import create_structured_output_runnable
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_community.chat_models import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import GraphDocument
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from neo4j_generation.utils import (
    KnowledgeGraph,
    map_to_base_node,
    map_to_base_relationship,
)

graph = Neo4jGraph()


llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)


def add_graph_documents(params: Dict) -> str:
    """
    Process the given text to extract graph data and constructs a graph document from the extracted information.
    The constructed graph document is then added to the graph.

    Parameters:
    - input (Dict): The input text from which the information will be extracted to construct the graph.

    Returns:
    str: A confirmation message indicating the completion of the graph construction.
    """  # noqa: E501

    data = params["data"]
    text = params["context"]["input"]
    # Construct a graph document
    graph_document = GraphDocument(
        nodes=[map_to_base_node(node) for node in data.nodes],
        relationships=[map_to_base_relationship(rel) for rel in data.rels],
        source=Document(page_content=text),
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])
    return "Graph construction finished"


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
- **Allowed Node Labels:** {allow_nodes}
- **Allowed Relationship Types**: {allow_rels}
## 3. Handling Numerical Data and Dates
- Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
- **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
- **Property Format**: Properties must be in a key-value format.
- **Quotation Marks**: Never use escaped single or double quotes within property values.
- **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
## 4. Coreference Resolution
- **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"),
always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.
Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial.
## 5. Strict Compliance
Adhere to the rules strictly. Non-compliance will result in termination.
          """,  # noqa: E501
        ),
        (
            "human",
            "Use the given format to extract information from the "
            "following input: {input}",
        ),
        ("human", "Tip: Make sure to answer in the correct format"),
    ]
)

extract_chain = create_structured_output_runnable(KnowledgeGraph, llm, prompt)

chain = RunnableParallel(
    {"data": extract_chain, "context": RunnablePassthrough()}
) | RunnableLambda(add_graph_documents)
