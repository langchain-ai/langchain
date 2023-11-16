import os
from langchain.embeddings import OpenAIEmbeddings

EMBEDDINGS = OpenAIEmbeddings(mode="text-embedding-ada-002")
EMBEDDINGS_DIMENSIONS = 1536  # known size of text-embedding-ada-002

DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
if not DOCUGAMI_API_KEY:
    raise Exception("Please set the DOCUGAMI_API_KEY environment variable")

PINECONE_INDEX = os.environ.get("PINECONE_INDEX")
if not PINECONE_INDEX:
    raise Exception("Please set the PINECONE_INDEX environment variable")

# Lengths for the loader are in terms of characters, 1 token ~= 4 chars in English
# Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
MAX_CHUNK_TEXT_LENGTH = 1024 * 128  # ~32k tokens
MIN_CHUNK_TEXT_LENGTH = 256


INDEXING_LOCAL_STATE_PATH = os.environ.get(
    "INDEXING_LOCAL_STATE_PATH", "temp/indexing_local_state.pkl"
)
