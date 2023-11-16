from typing import Dict, List
from langchain.schema import Document

def get_parent_id_mappings(chunks: List[Document]) -> Dict[str, Document]:
    # Separate out unique child and parent chunks for small-to-big retrieval.

    parents: Dict[str, Document] = {}
    for chunk in chunks:
        if not chunk.parent:
            continue

        parent_id: str = chunk.parent.metadata["id"]

        # Set parent metadata on all child chunks
        chunk.metadata["doc_id"] = parent_id

        # Keep track of all unique parents (by ID)
        if parent_id not in parents:
            parents[parent_id] = chunk.parent

    return parents
