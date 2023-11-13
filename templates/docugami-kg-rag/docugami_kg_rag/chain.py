import os
import pickle

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.vectorstores.pinecone import Pinecone
from langchain.storage.in_memory import InMemoryStore

if os.environ.get("OPENAI_API_KEY", None) is None:
    raise Exception("Missing `OPENAI_API_KEY` environment variable.")

# Pinecone options (please see README for notes on how to run indexing)
if os.environ.get("PINECONE_API_KEY", None) is None:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

if os.environ.get("PINECONE_ENVIRONMENT", None) is None:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

DOCUGAMI_DOCSET_ID = "fi6vi49cmeac"

PINECONE_INDEX_NAME = (
    os.environ.get("PINECONE_INDEX", "langchain-docugami")
    + f"-{DOCUGAMI_DOCSET_ID}"
)

PARENT_DOC_STORE_PATH = os.environ.get(
    "PARENT_DOC_STORE_ROOT_PATH", "temp/parent_docs.pkl"
)

# LangSmith options (set for tracing)
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "")

llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
embeddings = OpenAIEmbeddings()

# Chunks are in the vector store, and full documents are in an inmemory store
chunk_vectorstore = Pinecone.from_existing_index(PINECONE_INDEX_NAME, embeddings)
with open(PARENT_DOC_STORE_PATH, "rb") as file:
    parent_docstore: InMemoryStore = pickle.load(file)

retriever = MultiVectorRetriever(
    vectorstore=chunk_vectorstore, docstore=parent_docstore, search_kwargs={"k": 1}
)

# RAG answer synthesis prompt
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are a helpful assistant that answers questions based only on provided context. Your provided context can include text or tables, "
    "and may also contain semantic XML markup. Pay attention the semantic XML markup to understand more about the context semantics as "
    "well as structure (e.g. lists and tabular layouts expressed with HTML-like tags)"
)

human_prompt = HumanMessagePromptTemplate.from_template(
    """**** START CONTEXT:

{context}
**** END CONTEXT

Question: {question}"""
)

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        system_prompt,
        human_prompt,
    ]
)


chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | RAG_PROMPT
    | llm
    | StrOutputParser()
)
