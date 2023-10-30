import os

from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Weaviate

from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever

if os.environ.get("WEAVIATE_API_KEY", None) is None:
    raise Exception("Missing `WEAVIATE_API_KEY` environment variable.")

if os.environ.get("WEAVIATE_ENVIRONMENT", None) is None:
    raise Exception("Missing `WEAVIATE_ENVIRONMENT` environment variable.")

WEAVIATE_INDEX_NAME = os.environ.get("WEAVIATE_INDEX", "langchain-test")

### Ingest code - you may need to run this the first time
# Load
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# # Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# # Build retriever 
retriever = WeaviateHybridSearchRetriever = (
    client=client, index_name=WEAVIATE_INDEX_NAME, text_key="text", attributes=[], create_schema_if_missing=True
)

# # Add to vectorDB
retriever.add_documents(all_splits)


# Hybrid search
retriever.get_relevant_documents("agents short-term memory")

# Hybrid search with scores
retriever.get_relevant_documents("short-term memory", score=True)


