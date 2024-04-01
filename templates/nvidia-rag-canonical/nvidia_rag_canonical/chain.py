import getpass
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Milvus
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_nvidia_aiplay import ChatNVIDIA, NVIDIAEmbeddings
from langchain_text_splitters.character import CharacterTextSplitter

EMBEDDING_MODEL = "nvolveqa_40k"
CHAT_MODEL = "llama2_13b"
HOST = "127.0.0.1"
PORT = "19530"
COLLECTION_NAME = "test"
INGESTION_CHUNK_SIZE = 500
INGESTION_CHUNK_OVERLAP = 0

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")  # noqa: T201
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key

# Read from Milvus Vector Store
embeddings = NVIDIAEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Milvus(
    connection_args={"host": HOST, "port": PORT},
    collection_name=COLLECTION_NAME,
    embedding_function=embeddings,
)
retriever = vectorstore.as_retriever()

# RAG prompt
template = """<s>[INST] <<SYS>>
Use the following context to answer the user's question. If you don't know the answer,
just say that you don't know, don't try to make up an answer.
<</SYS>>
<s>[INST] Context: {context} Question: {question} Only return the helpful
 answer below and nothing else. Helpful answer:[/INST]"
"""
prompt = ChatPromptTemplate.from_template(template)

# RAG
model = ChatNVIDIA(model=CHAT_MODEL)
chain = (
    RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)


def _ingest(url: str) -> dict:
    """Load and ingest the PDF file from the URL"""

    loader = PyPDFLoader(url)
    data = loader.load()

    # Split docs
    text_splitter = CharacterTextSplitter(
        chunk_size=INGESTION_CHUNK_SIZE, chunk_overlap=INGESTION_CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(data)

    # Insert the documents in Milvus Vector Store
    _ = Milvus.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection_args={"host": HOST, "port": PORT},
    )
    return {}


ingest = RunnableLambda(_ingest)
