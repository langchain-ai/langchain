import os
import uuid

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymongo import MongoClient

PARENT_DOC_ID_KEY = "parent_doc_id"


def parent_child_splitter(data, id_key=PARENT_DOC_ID_KEY):
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # This text splitter is used to create the child documents
    # It should create documents smaller than the parent
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    documents = parent_splitter.split_documents(data)
    doc_ids = [str(uuid.uuid4()) for _ in documents]

    docs = []
    for i, doc in enumerate(documents):
        _id = doc_ids[i]
        sub_docs = child_splitter.split_documents([doc])
        for _doc in sub_docs:
            _doc.metadata[id_key] = _id
            _doc.metadata["doc_level"] = "child"
        docs.extend(sub_docs)
        doc.metadata[id_key] = _id
        doc.metadata["doc_level"] = "parent"
    return documents, docs


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
    parent_docs, child_docs = parent_child_splitter(data)

    # Insert the documents in MongoDB Atlas Vector Search
    _ = MongoDBAtlasVectorSearch.from_documents(
        documents=parent_docs + child_docs,
        embedding=OpenAIEmbeddings(disallowed_special=()),
        collection=MONGODB_COLLECTION,
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )
