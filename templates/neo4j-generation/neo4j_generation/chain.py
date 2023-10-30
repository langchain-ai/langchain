from typing import List, Optional

from langchain.chains.openai_functions import (
    create_structured_output_chain,
)
from langchain.chat_models import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.graphs.graph_document import GraphDocument
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from neo4j_generation.utils import (
    KnowledgeGraph,
    map_to_base_node,
    map_to_base_relationship,
)

graph = Neo4jGraph()


llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)


def get_extraction_chain(
    allowed_nodes: Optional[List[str]] = None, allowed_rels: Optional[List[str]] = None
):
    """
    Constructs and returns an extraction chain for building a knowledge graph based on specified parameters.

    The function generates a chat prompt template, outlining the instructions for an LLM to extract information
    and construct a knowledge graph. It primarily focuses on consistency in labeling nodes, handling numerical data
    and dates, coreference resolution, and strict compliance with the provided rules.

    Parameters:
    - allowed_nodes (Optional[List[str]]): A list of node labels that are allowed to be used in the knowledge graph.
                                           If not provided, there won't be any specific restriction on node labels.
    - allowed_rels (Optional[List[str]]): A list of relationship types that are allowed in the knowledge graph.
                                         If not provided, there won't be any specific restriction on relationship types.
    """  # noqa: E501
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                f"""# Knowledge Graph Instructions for GPT-4
## 1. Overview
You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
- **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
- The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
## 2. Labeling Nodes
- **Consistency**: Ensure you use basic or elementary types for node labels.
  - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
- **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
{'- **Allowed Node Labels:**' + ", ".join(allowed_nodes) if allowed_nodes else ""}
{'- **Allowed Relationship Types**:' + ", ".join(allowed_rels) if allowed_rels else ""}
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
    return create_structured_output_chain(KnowledgeGraph, llm, prompt, verbose=False)


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
    # Extract graph data using OpenAI functions
    extract_chain = get_extraction_chain(allowed_nodes, allowed_relationships)
    data = extract_chain.run(text)
    # Construct a graph document
    graph_document = GraphDocument(
        nodes=[map_to_base_node(node) for node in data.nodes],
        relationships=[map_to_base_relationship(rel) for rel in data.rels],
        source=Document(page_content=text),
    )
    # Store information into a graph
    graph.add_graph_documents([graph_document])
    return "Graph construction finished"
