# Ingest Documents into a Zep Collection
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import FakeEmbeddings
from langchain_community.vectorstores.zep import CollectionConfig, ZepVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter

ZEP_API_URL = os.environ.get("ZEP_API_URL", "http://localhost:8000")
ZEP_API_KEY = os.environ.get("ZEP_API_KEY", None)
ZEP_COLLECTION_NAME = os.environ.get("ZEP_COLLECTION", "langchaintest")

collection_config = CollectionConfig(
    name=ZEP_COLLECTION_NAME,
    description="Zep collection for LangChain",
    metadata={},
    embedding_dimensions=1536,
    is_auto_embedded=True,
)

# Load
loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
data = loader.load()

# Split
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Add to vectorDB
vectorstore = ZepVectorStore.from_documents(
    documents=all_splits,
    collection_name=ZEP_COLLECTION_NAME,
    config=collection_config,
    api_url=ZEP_API_URL,
    api_key=ZEP_API_KEY,
    embedding=FakeEmbeddings(size=1),
)
