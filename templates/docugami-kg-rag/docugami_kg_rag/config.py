import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.globals import set_llm_cache
from langchain.storage.in_memory import InMemoryStore

LLM = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

EMBEDDINGS = OpenAIEmbeddings(model="text-embedding-ada-002")

DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
if not DOCUGAMI_API_KEY:
    raise Exception("Please set the DOCUGAMI_API_KEY environment variable")

CHROMA_DIRECTORY = "/tmp/chroma_db"
os.makedirs(Path(CHROMA_DIRECTORY).parent, exist_ok=True)

INDEXING_LOCAL_STATE_PATH = os.environ.get(
    "INDEXING_LOCAL_STATE_PATH", "/tmp/indexing_local_state.pkl"
)
os.makedirs(Path(INDEXING_LOCAL_STATE_PATH).parent, exist_ok=True)

INDEXING_LOCAL_REPORT_DBS_ROOT = os.environ.get(
    "INDEXING_LOCAL_REPORT_DBS_ROOT", "/tmp/report_dbs"
)
os.makedirs(Path(INDEXING_LOCAL_REPORT_DBS_ROOT).parent, exist_ok=True)

LOCAL_LLM_CACHE_DB_FILE = os.environ.get("LOCAL_LLM_CACHE", "/tmp/.langchain.db")
os.makedirs(Path(LOCAL_LLM_CACHE_DB_FILE).parent, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=LOCAL_LLM_CACHE_DB_FILE))


@dataclass
class ReportDetails:
    id: str
    """ID of report."""

    name: str
    """Name of report."""

    local_xlsx_path: Path
    """Local path to XLSX of the report."""

    retrieval_tool_function_name: str
    """Function name for retrieval tool e.g. sql_query_earnings_calls."""

    retrieval_tool_description: str
    """
    Description of retrieval tool e.g. Runs a SQL query over the REPORT_NAME report, 
    represented as the following SQL Table... etc."""


@dataclass
class LocalIndexState:
    parents_by_id: InMemoryStore
    """Mapping of ID to parent chunks."""

    doc_summaries_by_id: InMemoryStore
    """Mapping of ID to document summaries."""

    retrieval_tool_function_name: str
    """Function name for retrieval tool e.g. "search_earnings_calls."""

    retrieval_tool_description: str
    """Description of retrieval tool e.g. Searches for and returns chunks from earnings call documents."""

    reports: List[ReportDetails] = field(default_factory=list)
    """Details about any reports for this docset."""


# Lengths for the loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MAX_CHUNK_TEXT_LENGTH = 1024 * 128  # ~32k tokens
MIN_CHUNK_TEXT_LENGTH = 256
SUB_CHUNK_TABLES = False
INCLUDE_XML_TAGS = True
PARENT_HIERARCHY_LEVELS = 1000
RETRIEVER_K = 10

SMALL_FRAGMENT_MAX_TEXT_LENGTH = 1024 * 4
LARGE_FRAGMENT_MAX_TEXT_LENGTH = 1024 * 20

BATCH_SIZE = 5
