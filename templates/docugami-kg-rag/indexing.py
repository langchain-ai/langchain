## Ingest code - you may need to run this the first time
# Load
import os
from pathlib import Path
import pickle



from langchain.storage.in_memory import InMemoryStore
from langchain.vectorstores.pinecone import Pinecone

import pinecone

from docugami_kg_rag.config import PINECONE_INDEX

if __name__ == "__main__":


    DOCUGAMI_DOCSET_ID = "fi6vi49cmeac"

    PARENT_DOC_STORE_PATH = 


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
