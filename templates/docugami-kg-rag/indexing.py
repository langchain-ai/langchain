## Ingest code - you may need to run this the first time
# Load
import os
from pathlib import Path
import pickle

from langchain.document_loaders import DocugamiLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.storage.in_memory import InMemoryStore
from langchain.vectorstores.pinecone import Pinecone

import pinecone

if __name__ == "__main__":
    EMBEDDINGS = OpenAIEmbeddings()
    EMBEDDINGS_DIMENSIONS = 1536  # known size of text-embedding-ada-002

    DOCUGAMI_API_KEY = os.environ.get("DOCUGAMI_API_KEY")
    DOCUGAMI_DOCSET_ID = "fi6vi49cmeac"

    # Lengths for the loader are in terms of characters, 1 token ~= 4 chars in English
    # Reference: https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them
    MAX_TEXT_LENGTH = 1024 * 128  # ~32k tokens
    MIN_TEXT_LENGTH = 256

    PINECONE_INDEX_NAME = (
        os.environ.get("PINECONE_INDEX", "langchain-docugami")
        + f"-{DOCUGAMI_DOCSET_ID}"
    )

    PARENT_DOC_STORE_PATH = os.environ.get(
        "PARENT_DOC_STORE_ROOT_PATH", "temp/parent_docs.pkl"
    )

    loader = DocugamiLoader(
        docset_id=DOCUGAMI_DOCSET_ID,
        min_text_length=MIN_TEXT_LENGTH,
        max_text_length=MAX_TEXT_LENGTH,
        sub_chunk_tables=False,
        include_xml_tags=True,
        parent_hierarchy_levels=1000,  # essentially entire document, up to max which is very large
        include_project_metadata_in_doc_metadata=False,  # not used, so lighten the vector index
        include_project_metadata_in_page_content=True,  # ok to include in vectorstore chunks, not used in context
    )

    chunks = loader.load()

    # Separate out unique child and parent chunks for small-to-big retrieval
    parents = {}
    for chunk in chunks:
        if not chunk.parent:
            continue

        parent_id = chunk.parent.metadata["id"]

        # Set parent metadata on all child chunks
        chunk.metadata["doc_id"] = parent_id

        # Keep track of all unique parents (by ID)
        if parent_id not in parents:
            parents[parent_id] = chunk.parent.page_content

    # Populate vectorDB with child chunks
    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        # Create index if it does not exist
        print(f"Creating pinecone index {PINECONE_INDEX_NAME}...")
        pinecone.create_index(name=PINECONE_INDEX_NAME, dimension=EMBEDDINGS_DIMENSIONS)

        print(f"Done creating pinecone index {PINECONE_INDEX_NAME}, now embedding...")
        Pinecone.from_documents(
            documents=chunks,
            embedding=EMBEDDINGS,
            index_name=PINECONE_INDEX_NAME,
        )

        print(f"Done embedding documents to pinecode index {PINECONE_INDEX_NAME}!")
    else:
        raise Exception(
            f"Index {PINECONE_INDEX_NAME} already exists. Please delete it to re-index this docset or you will get duplicate chunks."
        )

    # Populate the doc store with parent chunks
    store = InMemoryStore()
    store.mset(parents.items())

    # Dump store to disk (used in chain to load parent docs)
    store_local_path = Path(PARENT_DOC_STORE_PATH)
    os.makedirs(os.path.dirname(store_local_path), exist_ok=True)
    with open(store_local_path, "wb") as file:
        pickle.dump(store, file)
