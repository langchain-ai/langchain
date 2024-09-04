import os

from langchain_community.vectorstores import Aerospike
from aerospike_vector_search.types import VectorDistanceMetric
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import format_document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.runnables import (
    RunnablePassthrough,
)
from langchain_core.prompts.prompt import PromptTemplate


def get_bool_env(name, default):
    env = os.environ.get(name)
    if env is None:
        return default
    env = env.lower()

    if env in ["true", "1"]:
        return True
    else:
        return False


# Define the config
class Config(object):
    OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
    AVS_HOST = os.environ.get("AVS_HOST") or "localhost"
    AVS_PORT = int(os.environ.get("AVS_PORT") or 5000)
    AVS_INDEX_NAME = os.environ.get("AVS_INDEX_NAME") or "langchain-rag"
    AVS_IS_LOADBALANCER = get_bool_env("AVS_IS_LOADBALANCER", True)
    AVS_NAMESPACE = os.environ.get("AVS_NAMESPACE") or "test"
    DATASOURCE = os.environ.get("DATASOURCE") or "https://aerospike.com/files/ebooks/aerospike-up-and-running-early-release3.pdf"


# Initialize Aerospike Vector Search admin client
from aerospike_vector_search import AdminClient, types

avs_admin_client = AdminClient(
    seeds=types.HostPort(
        host=Config.AVS_HOST,
        port=Config.AVS_PORT,
    ),
    is_loadbalancer=Config.AVS_IS_LOADBALANCER,
)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
MODEL_DIM = 384
MODEL_DISTANCE_CALC = VectorDistanceMetric.COSINE
VECTOR_KEY = "vector"

index_exists = False
# Check if the index already exists. If not, create it
for index in avs_admin_client.index_list():
    if (
        index["id"]["namespace"] == Config.AVS_NAMESPACE
        and index["id"]["name"] == Config.AVS_INDEX_NAME
    ):
        index_exists = True
        print(f"{Config.AVS_INDEX_NAME} already exists. Skipping creation")
        break

# Create the HNSW index in Aerospike Vector Search
if not index_exists:
    avs_admin_client.index_create(
        namespace=Config.AVS_NAMESPACE,
        name=Config.AVS_INDEX_NAME,
        dimensions=MODEL_DIM,
        vector_distance_metric=MODEL_DISTANCE_CALC,
        vector_field=VECTOR_KEY,
    )

avs_admin_client.close()

# Initialize Aerospike Vector Search client
from aerospike_vector_search import Client

avs_client = Client(
    seeds=types.HostPort(
        host=Config.AVS_HOST,
        port=Config.AVS_PORT,
    ),
    is_loadbalancer=Config.AVS_IS_LOADBALANCER,
)

# load documents
# you can comment this out if this is not the first time running this chain
from langchain_community.document_loaders import PyPDFLoader
# For this example we use a PDF of the Aerospike DB architecture whitepaper.
# This RAG application will better answer Aerospike related questions based on this information.
loader = PyPDFLoader(Config.DATASOURCE)
data = loader.load()

from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=750, chunk_overlap=0)
documents = text_splitter.split_documents(data)


# Initialize Aerospike Vector Store
store = Aerospike(
    namespace=Config.AVS_NAMESPACE,
    client=avs_client,
    embedding=EMBEDDING,
    index_name=Config.AVS_INDEX_NAME,
    distance_strategy=MODEL_DISTANCE_CALC,
    vector_key=VECTOR_KEY,
)

# Ingest documents
# you can comment this out if this is not the first time running this chain
store.add_documents(documents)

retriever = store.as_retriever(k=8)
template = """Answer the question based on the following context:
Context: {context}
Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# Conversational Retrieval Chain
document_prompt = PromptTemplate.from_template(template="{page_content}")

# Combine documents returned by the Aerospike Vector Search retriever
# into a single string for better processing by the LLM
def _combine_documents(
    docs, document_prompt=document_prompt, document_separator="\n\n"
):
    doc_strings = [format_document(doc, document_prompt) for doc in docs]
    return document_separator.join(doc_strings)


# RAG
# This model requires that the OPENAI_API_KEY environment variable is set
model = ChatOpenAI()
chain = (
    {
        "context": retriever | _combine_documents,
        "question": RunnablePassthrough(),
    }
    | prompt
    | model
    | StrOutputParser()
)


# Add typing for input
class Question(BaseModel):
    __root__: str


chain = chain.with_types(input_type=Question)
