from typing import Dict, List
import json
import hashlib

from dataclasses import dataclass

# from langchain.document_manager.base import DocumentManager, ChunkOperation, DocumentWithOperation
from base import DocumentManager, ChunkOperation, DocumentWithOperation

from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter


def _get_hash(document: Document) -> str:
    """Returns the hash of the document."""
    hashable_content = document.page_content + json.dumps(document.metadata)
    return hashlib.sha256(hashable_content.encode('utf-8')).hexdigest()

def _get_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Returns the chunk id."""
    return f'{doc_id}-{chunk_index}'

@dataclass
class _DocumentWithState:
    """A document with its state."""
    document_id: str
    hash: str
    chunk_hashes: List[str]
    document: Document  


class InMemoryDocumentManager(DocumentManager):
    """In memory implementation of the document manager."""

    def __init__(self, text_splitter: TextSplitter):
        self._documents_with_state = {}
        self.text_splitter = text_splitter

    def add_documents_to_vectorstore(self,vectorstore) -> None:
        """Add documents to the vectorstore given their ids."""
        for doc,doc_hash in self.lazy_load_all_docs():
            vectorstore.add_texts(texts=[doc.page_content],metadatas=[doc.metadata],ids=[doc_hash])

    def lazy_load_all_docs(self,):     
        """Retrieve all documents."""
        for doc_id in self._documents_with_state:
            yield (self._documents_with_state[doc_id].document,
                   self._documents_with_state[doc_id].hash)

    def add(self, documents: List[Document], ids: List[str]) -> List[DocumentWithOperation]:
        """Adds documents to the document manager.

        Returns a Dict of chunks that should be added to the vector store.
        """
        if len(documents) != len(ids):
            raise ValueError("Unequal count of documents and ids.")

        seen = set()
        for id in ids:
            if id in seen:
                raise ValueError("Duplicate ids.")
            if id in self._documents_with_state:
                raise ValueError("Document with id {} already exists.".format(id))
            seen.add(id)

        chunks_to_add = []
        for document, id in zip(documents, ids):
            chunks = self.text_splitter.create_documents(document.page_content)
            chunk_hashes = [_get_hash(chunk) for chunk in chunks]
            # Store document in memory (for now)
            # As discussed, this would ideally be in a local key-value store
            doc_with_state = _DocumentWithState(id, self.get_document_hash(document), chunk_hashes, document)
            self._documents_with_state[id] = doc_with_state
            for i, chunk in enumerate(chunks):
                chunks_to_add.append(DocumentWithOperation(
                page_content=chunk.page_content, operation=ChunkOperation.ADD, id=_get_chunk_id(id, i)))
        return chunks_to_add



    def update(self, documents: List[Document], ids: List[str]) -> List[DocumentWithOperation]:
        """Updates documents in the document manager."""
        chunk_operations = []
        for new_document, id in zip(documents, ids):
            if id not in self._documents_with_state:
                raise ValueError(f"Document with id {id} does not exist.")
            doc_with_state = self._documents_with_state[id]
            new_document_hash = self.get_document_hash(new_document)
            if doc_with_state.hash != new_document_hash:
                new_chunks = self.text_splitter.create_documents(new_document.page_content)
                new_hashes = [_get_hash(chunk) for chunk in new_chunks]
                for i, new_chunk in enumerate(new_chunks):
                    if i < len(doc_with_state.chunk_hashes):
                        old_hash = doc_with_state.chunk_hashes[i]
                        if new_hashes[i] != old_hash:
                            chunk_operations.append(DocumentWithOperation(
                            page_content=new_chunk.page_content,
                            operation=ChunkOperation.UPDATE, id=_get_chunk_id(id, i)))
                    else:
                        # Add new chunks.
                        chunk_operations.append(DocumentWithOperation(
                            page_content=new_chunk.page_content,
                            operation=ChunkOperation.ADD, id=_get_chunk_id(id, i)))

                # If there are fewer new chunks than old chunks, remove the old chunks.
                if len(new_chunks) < len(doc_with_state.chunk_hashes):
                    # Remove old chunks.
                    for i in range(len(new_chunks), len(doc_with_state.chunk_hashes)):
                        # Add hacky placeholder value for page_content to satisfy pydantic.
                        chunk_operations.append(
                        DocumentWithOperation(page_content="placeholder", id=_get_chunk_id(id, i), operation=ChunkOperation.REMOVE))
                doc_with_state.hash = new_document_hash
                doc_with_state.chunk_hashes = new_hashes
        return chunk_operations


    def update_truncate(self, documents: List[Document], ids: List[str]):
        """Updates the documents in the document manager.
        Additionally, removes any documents that are not in
        `documents`.
        """
        chunk_operations = []
        if len(documents) != len(ids):
            raise ValueError("Unequal count of documents and ids.")
        id_set = set(ids)
        if len(ids) != len(id_set):
            raise ValueError("Duplicate ids.")

        # Compute chunks to be added and updated.
        for document, id in zip(documents, ids):
            if id not in self._documents_with_state:
                chunk_operations.extend(self.add([document], [id]))
            else:
                chunk_operations.extend(self.update([document], [id]))

        # Compute chunks to be deleted.
        for doc_id in self._documents_with_state:
            if doc_id not in id_set:
                doc_with_state = self._documents_with_state[doc_id]
                for i in range(len(doc_with_state.chunk_hashes)):
                    # Add hacky placeholder value for page_content to satisfy pydantic.
                    chunk_operations.append(
                        DocumentWithOperation(page_content="placeholder", id=_get_chunk_id(doc_id, i), operation=ChunkOperation.REMOVE))

        return chunk_operations
