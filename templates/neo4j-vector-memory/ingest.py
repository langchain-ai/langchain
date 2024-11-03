from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import TokenTextSplitter

txt_path = Path(__file__).parent / "dune.txt"

# Load the text file
loader = TextLoader(str(txt_path))
raw_documents = loader.load()

# Define chunking strategy
splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
documents = splitter.split_documents(raw_documents)

# Calculate embedding values and store them in the graph
Neo4jVector.from_documents(
    documents,
    OpenAIEmbeddings(),
    index_name="dune",
)
