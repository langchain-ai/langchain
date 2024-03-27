import hashlib
from typing import Dict, List, Optional

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.pydantic_v1 import root_validator
from langchain_core.retrievers import BaseRetriever

from langchain_community.vectorstores.chroma import Chroma


def hash_text(text: str) -> str:
    """Hash a text using SHA256.

    Args:
        text: Text to hash.

    Returns:
        Hashed text.
    """
    return str(hashlib.sha256(text.encode("utf-8")).hexdigest())

class ChromaRetriever(BaseRetriever):

    search_type: str = "similarity"

    store: Chroma

    class Config:
        """Configuration for this pydantic object."""

        arbitrary_types_allowed = True


    @root_validator()
    def validate_search_types(cls, values: Dict) -> Dict:
        # Check for the following search types as defined on the vector store interface
        # "max_marginal_relevance_search"
        # "similarity_search"
        assert values["search_type"] == "similarity" or values["search_type"] == "mmr"

        return values
    
    def update_document(
        self,
        document: Document,
        document_id: Optional[str] = None,
        **kwargs
    ) -> None:

        return self.store.update_document(
            document=document, 
            document_id=document_id, 
            **kwargs
        )

    def update_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> None:

        return self.store.update_documents(documents=documents, ids=ids, **kwargs)

    def add_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[dict]] = None,
        **kwargs
    ) -> None:

        return self.store.add_texts(texts=texts, metadatas=metadatas, ids=ids, **kwargs)
    
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        **kwargs
    ) -> None:

        return self.store.add_documents(documents=documents, ids=ids, **kwargs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs
    ) -> List[Document]:
        
        return self.store.search(query=query, search_type=self.search_type, **kwargs)
