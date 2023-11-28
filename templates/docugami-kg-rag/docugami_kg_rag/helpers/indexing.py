import os
import pickle
from pathlib import Path
from typing import Dict, List

import pinecone
from langchain.document_loaders import DocugamiLoader
from langchain.schema import Document
from langchain.storage.in_memory import InMemoryStore
from langchain.vectorstores.pinecone import Pinecone

from docugami_kg_rag.config import (
    EMBEDDINGS,
    EMBEDDINGS_DIMENSIONS,
    INCLUDE_XML_TAGS,
    INDEXING_LOCAL_STATE_PATH,
    MAX_CHUNK_TEXT_LENGTH,
    MIN_CHUNK_TEXT_LENGTH,
    PARENT_HIERARCHY_LEVELS,
    PINECONE_INDEX,
    SUB_CHUNK_TABLES,
    LocalIndexState,
)
from docugami_kg_rag.helpers.documents import build_summary_mappings
from docugami_kg_rag.helpers.reports import build_report_details
from docugami_kg_rag.helpers.retrieval import (
    chunks_to_direct_retriever_tool_description,
    docset_name_to_direct_retriever_tool_function_name,
)


def read_all_local_index_state() -> Dict[str, LocalIndexState]:
    if not Path(INDEXING_LOCAL_STATE_PATH).is_file():
        return {}  # not found

    with open(INDEXING_LOCAL_STATE_PATH, "rb") as file:
        return pickle.load(file)


def update_local_index(docset_id: str, name: str, parents_by_id: Dict[str, Document]):
    # Populate local index

    state = read_all_local_index_state()

    parents_by_id_store = InMemoryStore()
    parents_by_id_store.mset(list(parents_by_id.items()))

    doc_summaries = build_summary_mappings(parents_by_id)
    doc_summaries_by_id_store = InMemoryStore()
    doc_summaries_by_id_store.mset(list(doc_summaries.items()))

    direct_tool_function_name = docset_name_to_direct_retriever_tool_function_name(name)
    direct_tool_description = chunks_to_direct_retriever_tool_description(
        name, list(parents_by_id.values())
    )
    report_details = build_report_details(docset_id)

    doc_index_state = LocalIndexState(
        parents_by_id=parents_by_id_store,
        doc_summaries_by_id=doc_summaries_by_id_store,
        retrieval_tool_function_name=direct_tool_function_name,
        retrieval_tool_description=direct_tool_description,
        reports=report_details,
    )
    state[docset_id] = doc_index_state

    # Serialize state to disk (Deserialized in chain)
    store_local_path = Path(INDEXING_LOCAL_STATE_PATH)
    os.makedirs(os.path.dirname(store_local_path), exist_ok=True)
    with open(store_local_path, "wb") as file:
        pickle.dump(state, file)


def populate_pinecode_index(
    index_name: str, chunks: List[Document], force: bool = False
):
    # Populate pinecone with the given chunks

    if index_name in pinecone.list_indexes():
        if force:
            pinecone.delete_index(name=index_name)
        else:
            print(
                f"Reusing existing index {index_name} (use --force option to recreate)"
            )
            return

    # Create index if it does not exist
    print(f"Creating pinecone index {index_name}...")
    pinecone.create_index(name=index_name, dimension=EMBEDDINGS_DIMENSIONS)

    print(f"Done creating pinecone index {index_name}, now embedding...")
    Pinecone.from_documents(
        documents=chunks,
        embedding=EMBEDDINGS,
        index_name=index_name,
    )

    print(f"Done embedding documents to pinecode index {index_name}!")


def index_docset(docset_id: str, name: str, force: bool = False):
    # Indexes the given docset

    print(f"Indexing {name} (ID: {docset_id})")

    loader = DocugamiLoader(
        docset_id=docset_id,
        file_paths=None,
        document_ids=None,
        min_text_length=MIN_CHUNK_TEXT_LENGTH,
        max_text_length=MAX_CHUNK_TEXT_LENGTH,  # type: ignore
        sub_chunk_tables=SUB_CHUNK_TABLES,
        include_xml_tags=INCLUDE_XML_TAGS,
        parent_hierarchy_levels=PARENT_HIERARCHY_LEVELS,
        include_project_metadata_in_doc_metadata=False,  # not used, so lighten the vector index
    )

    chunks = loader.load()

    # Build separate maps of parent and child chunks
    parents_by_id: Dict[str, Document] = {}
    children_by_id: Dict[str, Document] = {}
    for chunk in chunks:
        chunk_id = chunk.metadata.get("id")
        parent_chunk_id = chunk.metadata.get(loader.parent_id_key)
        if not parent_chunk_id:
            # parent chunk
            parents_by_id[chunk_id] = chunk
        else:
            # child chunk
            children_by_id[chunk_id] = chunk

    docset_pinecone_index_name = f"{PINECONE_INDEX}-{docset_id}"
    populate_pinecode_index(
        docset_pinecone_index_name, list(children_by_id.values()), force
    )
    update_local_index(docset_id, name, parents_by_id)
