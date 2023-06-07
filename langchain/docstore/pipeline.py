"""Module implements a pipeline.

There might be a better name for this.
"""
from __future__ import annotations

import datetime
from typing import Sequence, Optional, Iterator, Iterable, List

from langchain.docstore.base import ArtifactWithData, ArtifactStore, Selector
from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document, BaseDocumentTransformer
from langchain.text_splitter import TextSplitter


def _convert_document_to_artifact_upsert(
    document: Document, parent_documents: Sequence[Document], transformation_hash: str
) -> ArtifactWithData:
    """Convert the given documents to artifacts for upserting."""
    dt = datetime.datetime.now().isoformat()
    parent_uids = [str(parent_doc.uid) for parent_doc in parent_documents]
    parent_hashes = [str(parent_doc.hash_) for parent_doc in parent_documents]

    return {
        "artifact": {
            "uid": str(document.uid),
            "parent_uids": parent_uids,
            "metadata": document.metadata,
            "parent_hashes": parent_hashes,
            "tags": tuple(),
            "type_": "document",
            "data": None,
            "location": None,
            "data_hash": str(document.hash_),
            "metadata_hash": "N/A",
            "created_at": dt,
            "updated_at": dt,
            "transformation_hash": transformation_hash,
        },
        "document": document,
    }


class Pipeline(BaseLoader):  # MAY NOT WANT TO INHERIT FROM LOADER
    def __init__(
        self,
        loader: BaseLoader,
        *,
        transformers: Optional[Sequence[BaseDocumentTransformer]] = None,
        artifact_store: Optional[ArtifactStore] = None,
    ) -> None:
        """Initialize the document pipeline.

        Args:
            loader: The loader to use for loading the documents.
            transformers: The transformers to use for transforming the documents.
            artifact_store: The artifact store to use for storing the artifacts.
        """
        self.loader = loader
        self.transformers = transformers
        self.artifact_store = artifact_store

    def lazy_load(
        self,
    ) -> Iterator[Document]:
        """Lazy load the documents."""
        transformations = self.transformers or []
        # Need syntax for determining whether this should be cached.

        try:
            doc_iterator = self.loader.lazy_load()
        except NotImplementedError:
            doc_iterator = self.loader.load()

        for document in doc_iterator:
            new_documents = [document]
            for transformation in transformations:
                # Batched for now here -- lots of optimization possible
                # but not needed for now and is likely going to get complex
                new_documents = list(
                    self._propagate_documents(new_documents, transformation)
                )

            yield from new_documents

    def _propagate_documents(
        self, documents: Sequence[Document], transformation: BaseDocumentTransformer
    ) -> Iterable[Document]:
        """Transform the given documents using the transformation with caching."""
        docs_exist = self.artifact_store.exists_by_uid(
            [document.uid for document in documents]
        )

        for document, exists in zip(documents, docs_exist):
            if exists:
                existing_docs = self.artifact_store.list_documents(
                    Selector(parent_uids=[document.uid])
                )

                materialized_docs = list(existing_docs)

                if materialized_docs:
                    yield from materialized_docs
                    continue

            transformed_docs = transformation.transform_documents([document])

            # MAJOR: Hash should encapsulate transformation parameters
            transformation_hash = transformation.__class__.__name__

            artifacts_with_data = [
                _convert_document_to_artifact_upsert(
                    transformed_doc, [document], transformation_hash
                )
                for transformed_doc in transformed_docs
            ]

            self.artifact_store.upsert(artifacts_with_data)
            yield from transformed_docs

    def load(self) -> List[Document]:
        """Load the documents."""
        return list(self.lazy_load())

    def run(self) -> None:  # BAD API NEED
        """Execute the pipeline, returning nothing."""
        for _ in self.lazy_load():
            pass

    def load_and_split(
        self, text_splitter: Optional[TextSplitter] = None
    ) -> List[Document]:
        raise NotImplementedError("This method will never be implemented.")
