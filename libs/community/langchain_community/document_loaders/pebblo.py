"""Pebblo's safe dataloader is a wrapper for document loaders"""

import logging
import os
import uuid
from importlib.metadata import version
from typing import Dict, Iterator, List, Optional

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

    @classmethod
    def set_loader_sent(cls) -> None:
        cls._loader_sent = True

    def _classify_doc(self, loaded_docs: list, loading_end: bool = False) -> list:
        """Send documents fetched from loader to pebblo-server. Then send
        classified documents to Daxa cloud(If api_key is present). Internal method.

        Args:

            loaded_docs (list): List of documents fetched from loader's load operation.
            loading_end (bool, optional): Flag indicating the halt of data
                                          loading by loader. Defaults to False.
        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if loading_end is True:
            PebbloSafeLoader.set_loader_sent()
        doc_content = [doc.dict() for doc in loaded_docs]
        docs = []
        classified_docs = []
        for doc in doc_content:
            doc_metadata = doc.get("metadata", {})
            doc_authorized_identities = doc_metadata.get("authorized_identities", [])
            doc_source_path = get_full_path(
                doc_metadata.get(
                    "full_path", doc_metadata.get("source", self.source_path)
                )
            )
            doc_source_owner = doc_metadata.get(
                "owner", PebbloSafeLoader.get_file_owner_from_path(doc_source_path)
            )
            doc_source_size = doc_metadata.get(
                "size", self.get_source_size(doc_source_path)
            )
            page_content = str(doc.get("page_content"))
            page_content_size = self.calculate_content_size(page_content)
            self.source_aggregate_size += page_content_size
            doc_id = doc.get("id", None) or 0
            docs.append(
                {
                    "doc": page_content,
                    "source_path": doc_source_path,
                    "id": doc_id,
                    "last_modified": doc.get("metadata", {}).get("last_modified"),
                    "file_owner": doc_source_owner,
                    **(
                        {"authorized_identities": doc_authorized_identities}
                        if doc_authorized_identities
                        else {}
                    ),
                    **(
                        {"source_path_size": doc_source_size}
                        if doc_source_size is not None
                        else {}
                    ),
                }
            )
        payload: Dict[str, Any] = {
            "name": self.app_name,
            "owner": self.owner,
            "docs": docs,
            "plugin_version": PLUGIN_VERSION,
            "load_id": self.load_id,
            "loader_details": self.loader_details,
            "loading_end": "false",
            "source_owner": self.source_owner,
            "classifier_location": self.classifier_location,
        }
        if loading_end is True:
            payload["loading_end"] = "true"
            if "loader_details" in payload:
                payload["loader_details"]["source_aggregate_size"] = (
                    self.source_aggregate_size
                )
        payload = Doc(**payload).dict(exclude_unset=True)
        # Raw payload to be sent to classifier
        if self.classifier_location == "local":
            load_doc_url = f"{self.classifier_url}{LOADER_DOC_URL}"
            try:
                pebblo_resp = requests.post(
                    load_doc_url, headers=headers, json=payload, timeout=300
                )
                classified_docs = json.loads(pebblo_resp.text).get("docs", None)
                if pebblo_resp.status_code not in [
                    HTTPStatus.OK,
                    HTTPStatus.BAD_GATEWAY,
                ]:
                    logger.warning(
                        "Received unexpected HTTP response code: %s",
                        pebblo_resp.status_code,
                    )
                logger.debug(
                    "send_loader_doc[local]: request url %s, body %s len %s\
                        response status %s body %s",
                    pebblo_resp.request.url,
                    str(pebblo_resp.request.body),
                    str(
                        len(
                            pebblo_resp.request.body if pebblo_resp.request.body else []
                        )
                    ),
                    str(pebblo_resp.status_code),
                    pebblo_resp.json(),
                )
            except requests.exceptions.RequestException:
                logger.warning("Unable to reach pebblo server.")
            except Exception as e:
                logger.warning("An Exception caught in _send_loader_doc: local %s", e)

        if self.api_key:
            if self.classifier_location == "local":
                payload["docs"] = classified_docs
            headers.update({"x-api-key": self.api_key})
            pebblo_cloud_url = f"{PEBBLO_CLOUD_URL}{LOADER_DOC_URL}"
            try:
                pebblo_cloud_response = requests.post(
                    pebblo_cloud_url, headers=headers, json=payload, timeout=20
                )
                logger.debug(
                    "send_loader_doc[cloud]: request url %s, body %s len %s\
                        response status %s body %s",
                    pebblo_cloud_response.request.url,
                    str(pebblo_cloud_response.request.body),
                    str(
                        len(
                            pebblo_cloud_response.request.body
                            if pebblo_cloud_response.request.body
                            else []
                        )
                    ),
                    str(pebblo_cloud_response.status_code),
                    pebblo_cloud_response.json(),
                )
            except requests.exceptions.RequestException:
                logger.warning("Unable to reach Pebblo cloud server.")
            except Exception as e:
                logger.warning("An Exception caught in _send_loader_doc: cloud %s", e)
        elif self.classifier_location == "pebblo-cloud":
            logger.warning("API key is missing for sending docs to Pebblo cloud.")
            raise NameError("API key is missing for sending docs to Pebblo cloud.")

        return classified_docs

    @staticmethod
    def calculate_content_size(page_content: str) -> int:
        """Calculate the content size in bytes:
        - Encode the string to bytes using a specific encoding (e.g., UTF-8)
        - Get the length of the encoded bytes.

        Args:
            page_content (str): Data string.

        Returns:
            int: Size of string in bytes.
        """

        # Encode the content to bytes using UTF-8
        encoded_content = page_content.encode("utf-8")
        size = len(encoded_content)
        return size

    def _send_discover(self) -> None:
        """Send app discovery payload to pebblo-server. Internal method."""
        pebblo_resp = None
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = self.app.dict(exclude_unset=True)
        # Raw discover payload to be sent to classifier
        if self.classifier_location == "local":
            app_discover_url = f"{self.classifier_url}{APP_DISCOVER_URL}"
            try:
                pebblo_resp = requests.post(
                    app_discover_url, headers=headers, json=payload, timeout=20
                )
                logger.debug(
                    "send_discover[local]: request url %s, body %s len %s\
                        response status %s body %s",
                    pebblo_resp.request.url,
                    str(pebblo_resp.request.body),
                    str(
                        len(
                            pebblo_resp.request.body if pebblo_resp.request.body else []
                        )
                    ),
                    str(pebblo_resp.status_code),
                    pebblo_resp.json(),
                )
                if pebblo_resp.status_code in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
                    PebbloSafeLoader.set_discover_sent()
                else:
                    logger.warning(
                        f"Received unexpected HTTP response code:\
                            {pebblo_resp.status_code}"
                    )
            except requests.exceptions.RequestException:
                logger.warning("Unable to reach pebblo server.")
            except Exception as e:
                logger.warning("An Exception caught in _send_discover: local %s", e)

        if self.api_key:
            try:
                headers.update({"x-api-key": self.api_key})
                # If the pebblo_resp is None,
                # then the pebblo server version is not available
                if pebblo_resp:
                    pebblo_server_version = json.loads(pebblo_resp.text).get(
                        "pebblo_server_version"
                    )
                    payload.update({"pebblo_server_version": pebblo_server_version})

                payload.update({"pebblo_client_version": PLUGIN_VERSION})
                pebblo_cloud_url = f"{PEBBLO_CLOUD_URL}{APP_DISCOVER_URL}"
                pebblo_cloud_response = requests.post(
                    pebblo_cloud_url, headers=headers, json=payload, timeout=20
                )

                logger.debug(
                    "send_discover[cloud]: request url %s, body %s len %s\
                        response status %s body %s",
                    pebblo_cloud_response.request.url,
                    str(pebblo_cloud_response.request.body),
                    str(
                        len(
                            pebblo_cloud_response.request.body
                            if pebblo_cloud_response.request.body
                            else []
                        )
                    ),
                    str(pebblo_cloud_response.status_code),
                    pebblo_cloud_response.json(),
                )
            except requests.exceptions.RequestException:
                logger.warning("Unable to reach Pebblo cloud server.")
            except Exception as e:
                logger.warning("An Exception caught in _send_discover: cloud %s", e)

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
