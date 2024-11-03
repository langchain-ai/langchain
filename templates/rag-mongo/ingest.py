import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

MONGO_URI = os.environ["MONGO_URI"]

# Note that if you change this, you also need to change it in `rag_mongo/chain.py`
DB_NAME = "langchain-test-2"
COLLECTION_NAME = "test"
ATLAS_VECTOR_SEARCH_INDEX_NAME = "default"
EMBEDDING_FIELD_NAME = "embedding"
client = MongoClient(MONGO_URI)
db = client[DB_NAME]
MONGODB_COLLECTION = db[COLLECTION_NAME]

if __name__ == "__main__":
    # Load docs
    loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
    data = loader.load()

    # Split docs
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    docs = text_splitter.split_documents(data)

    # Insert the documents in MongoDB Atlas Vector Search
    _ = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
