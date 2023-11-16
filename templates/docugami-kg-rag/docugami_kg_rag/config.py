from dataclasses import dataclass
import os

from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage.in_memory import InMemoryStore

LLM = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")
EMBEDDINGS_DIMENSIONS = 1536  # known size of text-embedding-ada-002

DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
if not DOCUGAMI_API_KEY:
    raise Exception("Please set the DOCUGAMI_API_KEY environment variable")

PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
if not PINECONE_INDEX:
    raise Exception("Please set the PINECONE_INDEX environment variable")

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise Exception("Missing `PINECONE_API_KEY` environment variable.")

PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
if not PINECONE_ENVIRONMENT:
    raise Exception("Missing `PINECONE_ENVIRONMENT` environment variable.")

INDEXING_LOCAL_STATE_PATH = os.environ.get("INDEXING_LOCAL_STATE_PATH", "temp/indexing_local_state.pkl")


@dataclass
class LocalIndexState:
    parents_by_id: InMemoryStore
    retrieval_tool_function_name: str
    """e.g. "search_earnings_calls"""

    retrieval_tool_description: str
    """e.g. Searches for and returns chunks from earnings call documents."""


# Lengths for the loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MAX_CHUNK_TEXT_LENGTH = 1024 * 128  # ~32k tokens
MIN_CHUNK_TEXT_LENGTH = 256
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = False
PARENT_HIERARCHY_LEVELS = 1000
RETRIEVER_K = 6

# LangSmith options (set for tracing)
LANGCHAIN_TRACING_V2 = True
LANGCHAIN_ENDPOINT = "https://api.smith.langchain.com"
LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", "")
LANGCHAIN_PROJECT = os.environ.get("LANGCHAIN_PROJECT", "")
