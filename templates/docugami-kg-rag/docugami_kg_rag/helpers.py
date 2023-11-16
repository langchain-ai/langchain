
from langchain.document_loaders import DocugamiLoader

from docugami_kg_rag.config import MAX_CHUNK_TEXT_LENGTH, MIN_CHUNK_TEXT_LENGTH, PINECONE_INDEX


def index_docset(id: str, name: str):
    print(f"Indexing {name} (ID: {id})")

    loader = DocugamiLoader(
        docset_id=id,
        min_text_length=MIN_CHUNK_TEXT_LENGTH,
        max_text_length=MAX_CHUNK_TEXT_LENGTH,
        sub_chunk_tables=False,
        include_xml_tags=True,
        parent_hierarchy_levels=1000,  # essentially entire document, up to max which is very large
        include_project_metadata_in_doc_metadata=False,  # not used, so lighten the vector index
        include_project_metadata_in_page_content=True,  # ok to include in vectorstore chunks, not used in context
    )

    pinecone_index_name = f"{PINECONE_INDEX}-{id}"
