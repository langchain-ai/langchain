"""Pebblo's safe dataloader is a wrapper for document loaders"""

import logging
import os
import uuid
from importlib.metadata import version
from typing import Any, Dict, Iterable, Iterator, List, Optional

from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.pebblo import (
    BATCH_SIZE_BYTES,
    PLUGIN_VERSION,
    App,
    Framework,
    IndexedDocument,
    PebbloLoaderAPIWrapper,
    generate_size_based_batches,
    get_full_path,
    get_loader_full_path,
    get_loader_type,
    get_runtime,
    get_source_size,
)

logger = logging.getLogger(__name__)


class PebbloSafeLoader(BaseLoader):
    """Pebblo Safe Loader class is a wrapper around document loaders enabling the data
    to be scrutinized.
    """

    _discover_sent: bool = False

    def __init__(
        self,
        langchain_loader: BaseLoader,
        name: str,
        owner: str = "",
        description: str = "",
        api_key: Optional[str] = None,
        load_semantic: bool = False,
        classifier_url: Optional[str] = None,
        *,
        classifier_location: str = "local",
        anonymize_snippets: bool = False,
    ):
        if not name or not isinstance(name, str):
            raise NameError("Must specify a valid name.")
        self.app_name = name
        self.load_id = str(uuid.uuid4())
        self.loader = langchain_loader
        self.load_semantic = os.environ.get("PEBBLO_LOAD_SEMANTIC") or load_semantic
        self.owner = owner
        self.description = description
        self.source_path = get_loader_full_path(self.loader)
        self.docs: List[Document] = []
        self.docs_with_id: List[IndexedDocument] = []
        loader_name = str(type(self.loader)).split(".")[-1].split("'")[0]
        self.source_type = get_loader_type(loader_name)
        self.source_path_size = get_source_size(self.source_path)
        self.batch_size = BATCH_SIZE_BYTES
        self.loader_details = {
            "loader": loader_name,
            "source_path": self.source_path,
            "source_type": self.source_type,
            **(
                {"source_path_size": str(self.source_path_size)}
                if self.source_path_size > 0
                else {}
            ),
        }
        # generate app
        self.app = self._get_app_details()
        # initialize Pebblo Loader API client
        self.pb_client = PebbloLoaderAPIWrapper(
            api_key=api_key,
            classifier_location=classifier_location,
            classifier_url=classifier_url,
            anonymize_snippets=anonymize_snippets,
        )
        self.pb_client.send_loader_discover(self.app)

    def load(self) -> List[Document]:
        """Load Documents.

        Returns:
            list: Documents fetched from load method of the wrapped `loader`.
        """
        self.docs = self.loader.load()
        # Classify docs in batches
        self.classify_in_batches()
        return self.docs

    def classify_in_batches(self) -> None:
        """
        Classify documents in batches.
        This is to avoid API timeouts when sending large number of documents.
        Batches are generated based on the page_content size.
        """
        batches: List[List[Document]] = generate_size_based_batches(
            self.docs, self.batch_size
        )

        processed_docs: List[Document] = []

        total_batches = len(batches)
        for i, batch in enumerate(batches):
            is_last_batch: bool = i == total_batches - 1
            self.docs = batch
            self.docs_with_id = self._index_docs()
            classified_docs = self.pb_client.classify_documents(
                self.docs_with_id,
                self.app,
                self.loader_details,
                loading_end=is_last_batch,
            )
            self._add_pebblo_specific_metadata(classified_docs)
            if self.load_semantic:
                batch_processed_docs = self._add_semantic_to_docs(classified_docs)
            else:
                batch_processed_docs = self._unindex_docs()
            processed_docs.extend(batch_processed_docs)

        self.docs = processed_docs

    def lazy_load(self) -> Iterator[Document]:
        """Load documents in lazy fashion.

        Raises:
            NotImplementedError: raised when lazy_load id not implemented
            within wrapped loader.

        Yields:
            list: Documents from loader's lazy loading.
        """
        try:
            doc_iterator = self.loader.lazy_load()
        except NotImplementedError as exc:
            err_str = f"{self.loader.__class__.__name__} does not implement lazy_load()"
            logger.error(err_str)
            raise NotImplementedError(err_str) from exc
        while True:
            try:
                doc = next(doc_iterator)
            except StopIteration:
                self.docs = []
                break
            self.docs = list((doc,))
            self.docs_with_id = self._index_docs()
            classified_doc = self.pb_client.classify_documents(
                self.docs_with_id, self.app, self.loader_details
            )
            self._add_pebblo_specific_metadata(classified_doc)
            if self.load_semantic:
                self.docs = self._add_semantic_to_docs(classified_doc)
            else:
                self.docs = self._unindex_docs()
            yield self.docs[0]

    @classmethod
    def set_discover_sent(cls) -> None:
        cls._discover_sent = True

    def _get_app_details(self) -> App:
        """Fetch app details. Internal method.

        Returns:
            App: App details.
        """
        framework, runtime = get_runtime()
        app = App(
            name=self.app_name,
            owner=self.owner,
            description=self.description,
            load_id=self.load_id,
            runtime=runtime,
            framework=framework,
            plugin_version=PLUGIN_VERSION,
            client_version=Framework(
                name="langchain_community",
                version=version("langchain_community"),
            ),
        )
        return app

    def _index_docs(self) -> List[IndexedDocument]:
        """
        Indexes the documents and returns a list of IndexedDocument objects.

        Returns:
            List[IndexedDocument]: A list of IndexedDocument objects with unique IDs.
        """
        docs_with_id = [
            IndexedDocument(pb_id=str(i), **doc.dict())
            for i, doc in enumerate(self.docs)
        ]
        return docs_with_id

    def _add_semantic_to_docs(self, classified_docs: Dict) -> List[Document]:
        """
        Adds semantic metadata to the given list of documents.

        Args:
            classified_docs (Dict): A dictionary of dictionaries containing the
                classified documents with pb_id as key.

        Returns:
            List[Document]: A list of Document objects with added semantic metadata.
        """
        indexed_docs = {
            doc.pb_id: Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in self.docs_with_id
        }

        for classified_doc in classified_docs.values():
            doc_id = classified_doc.get("pb_id")
            if doc_id in indexed_docs:
                self._add_semantic_to_doc(indexed_docs[doc_id], classified_doc)

        semantic_metadata_docs = [doc for doc in indexed_docs.values()]

        return semantic_metadata_docs

    def _unindex_docs(self) -> List[Document]:
        """
        Converts a list of IndexedDocument objects to a list of Document objects.

        Returns:
            List[Document]: A list of Document objects.
        """
        docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for i, doc in enumerate(self.docs_with_id)
        ]
        return docs

    def _add_semantic_to_doc(self, doc: Document, classified_doc: dict) -> Document:
        """
        Adds semantic metadata to the given document in-place.

        Args:
            doc (Document): A Document object.
            classified_doc (dict): A dictionary containing the classified document.

        Returns:
            Document: The Document object with added semantic metadata.
        """
        doc.metadata["pebblo_semantic_entities"] = list(
            classified_doc.get("entities", {}).keys()
        )
        doc.metadata["pebblo_semantic_topics"] = list(
            classified_doc.get("topics", {}).keys()
        )
        return doc

    def _add_pebblo_specific_metadata(self, classified_docs: dict) -> None:
        """Add Pebblo specific metadata to documents."""
        for doc in self.docs_with_id:
            doc_metadata = doc.metadata
            if self.loader.__class__.__name__ == "SharePointLoader":
                doc_metadata["full_path"] = get_full_path(
                    doc_metadata.get("source", self.source_path)
                )
            else:
                doc_metadata["full_path"] = get_full_path(
                    doc_metadata.get(
                        "full_path", doc_metadata.get("source", self.source_path)
                    )
                )
            doc_metadata["pb_checksum"] = classified_docs.get(doc.pb_id, {}).get(
                "pb_checksum", None
            )


class PebbloTextLoader(BaseLoader):
    """
    Loader for text data.

    Since PebbloSafeLoader is a wrapper around document loaders, this loader is
    used to load text data directly into Documents.
    """

    def __init__(
        self,
        texts: Iterable[str],
        *,
        source: Optional[str] = None,
        ids: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """
        Args:
            texts: Iterable of text data.
            source: Source of the text data.
                Optional. Defaults to None.
            ids: List of unique identifiers for each text.
                Optional. Defaults to None.
            metadata: Metadata for all texts.
                Optional. Defaults to None.
            metadatas: List of metadata for each text.
                Optional. Defaults to None.
        """
        self.texts = texts
        self.source = source
        self.ids = ids
        self.metadata = metadata
        self.metadatas = metadatas

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazy load text data into Documents.

        Returns:
            Iterator of Documents
        """
        for i, text in enumerate(self.texts):
            _id = None
            metadata = self.metadata or {}
            if self.metadatas and i < len(self.metadatas) and self.metadatas[i]:
                metadata.update(self.metadatas[i])
            if self.ids and i < len(self.ids):
                _id = self.ids[i]
            yield Document(id=_id, page_content=text, metadata=metadata)

    def load(self) -> List[Document]:
        """
        Load text data into Documents.

        Returns:
            List of Documents
        """
        documents = []
        for doc in self.lazy_load():
            documents.append(doc)
        return documents
