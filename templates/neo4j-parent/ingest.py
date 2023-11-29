from pathlib import Path

from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.graphs import Neo4jGraph
from langchain.text_splitter import TokenTextSplitter
from langchain.vectorstores import Neo4jVector

txt_path = Path(__file__).parent / "dune.txt"

graph = Neo4jGraph()

# Load the text file
loader = TextLoader(txt_path)
documents = loader.load()

# Define chunking strategy
parent_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
child_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=24)

# Store parent-child patterns into graph
parent_documents = parent_splitter.split_documents(documents)
for parent in parent_documents:
    child_documents = child_splitter.split_documents([parent])
    params = {
        "parent": parent.page_content,
        "children": [c.page_content for c in child_documents],
    }
    graph.query(
        """
    CREATE (p:Parent {text: $parent})
    WITH p 
    UNWIND $children AS child
    CREATE (c:Child {text: child})
    CREATE (c)-[:HAS_PARENT]->(p)
    """,
        params,
    )

# Calculate embedding values on the child nodes
Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    index_name="retrieval",
    node_label="Child",
    text_node_properties=["text"],
    embedding_node_property="embedding",
)
