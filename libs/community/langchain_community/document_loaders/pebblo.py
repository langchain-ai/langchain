"""Pebblo's safe dataloader is a wrapper for document loaders"""

import json
import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, Dict, Iterator, List, Optional, Union

import requests  # type: ignore
from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.pebblo import (
    APP_DISCOVER_URL,
    CLASSIFIER_URL,
    LOADER_DOC_URL,
    PEBBLO_CLOUD_URL,
    PLUGIN_VERSION,
    App,
    Doc,
    IndexedDocument,
    get_full_path,
    get_loader_full_path,
    get_loader_type,
    get_runtime,
)

logger = logging.getLogger(__name__)


class PebbloSafeLoader(BaseLoader):
    """Pebblo Safe Loader class is a wrapper around document loaders enabling the data
    to be scrutinized.
    """

    _discover_sent: bool = False
    _loader_sent: bool = False

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
        self.api_key = os.environ.get("PEBBLO_API_KEY") or api_key
        self.load_id = str(uuid.uuid4())
        self.loader = langchain_loader
        self.load_semantic = os.environ.get("PEBBLO_LOAD_SEMANTIC") or load_semantic
        self.owner = owner
        self.description = description
        self.source_path = get_loader_full_path(self.loader)
        self.source_owner = PebbloSafeLoader.get_file_owner_from_path(self.source_path)
        self.docs: List[Document] = []
        self.docs_with_id: Union[List[IndexedDocument], List[Document], List] = []
        loader_name = str(type(self.loader)).split(".")[-1].split("'")[0]
        self.source_type = get_loader_type(loader_name)
        self.source_path_size = self.get_source_size(self.source_path)
        self.source_aggregate_size = 0
        self.classifier_url = classifier_url or CLASSIFIER_URL
        self.classifier_location = classifier_location
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
        self._send_discover()

    def load(self) -> List[Document]:
        """Load Documents.

        Returns:
            list: Documents fetched from load method of the wrapped `loader`.
        """
        self.docs = self.loader.load()
        if not self.load_semantic:
            self._classify_doc(self.docs, loading_end=True)
            return self.docs
        self.docs_with_id = self._index_docs()
        classified_docs = self._classify_doc(self.docs_with_id, loading_end=True)
        self.docs_with_id = self._add_semantic_to_docs(
            self.docs_with_id, classified_docs
        )
        self.docs = self._unindex_docs(self.docs_with_id)  # type: ignore
        return self.docs

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
            if not self.load_semantic:
                self._classify_doc(self.docs, loading_end=True)
                yield self.docs[0]
            else:
                self.docs_with_id = self._index_docs()
                classified_doc = self._classify_doc(self.docs)
                self.docs_with_id = self._add_semantic_to_docs(
                    self.docs_with_id, classified_doc
                )
                self.docs = self._unindex_docs(self.docs_with_id)  # type: ignore
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
                payload["loader_details"][
                    "source_aggregate_size"
                ] = self.source_aggregate_size
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
        )
        return app

    @staticmethod
    def get_file_owner_from_path(file_path: str) -> str:
        """Fetch owner of local file path.

        Args:
            file_path (str): Local file path.

        Returns:
            str: Name of owner.
        """
        try:
            import pwd

            file_owner_uid = os.stat(file_path).st_uid
            file_owner_name = pwd.getpwuid(file_owner_uid).pw_name
        except Exception:
            file_owner_name = "unknown"
        return file_owner_name

    def get_source_size(self, source_path: str) -> int:
        """Fetch size of source path. Source can be a directory or a file.

        Args:
            source_path (str): Local path of data source.

        Returns:
            int: Source size in bytes.
        """
        if not source_path:
            return 0
        size = 0
        if os.path.isfile(source_path):
            size = os.path.getsize(source_path)
        elif os.path.isdir(source_path):
            total_size = 0
            for dirpath, _, filenames in os.walk(source_path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            size = total_size
        return size

    def _index_docs(self) -> List[IndexedDocument]:
        """
        Indexes the documents and returns a list of IndexedDocument objects.

        Returns:
            List[IndexedDocument]: A list of IndexedDocument objects with unique IDs.
        """
        docs_with_id = [
            IndexedDocument(id=hex(i)[2:], **doc.dict())
            for i, doc in enumerate(self.docs)
        ]
        return docs_with_id

    def _add_semantic_to_docs(
        self, docs_with_id: List[IndexedDocument], classified_docs: List[dict]
    ) -> List[Document]:
        """
        Adds semantic metadata to the given list of documents.

        Args:
            docs_with_id (List[IndexedDocument]): A list of IndexedDocument objects
                containing the documents with their IDs.
            classified_docs (List[dict]): A list of dictionaries containing the
                classified documents.

        Returns:
            List[Document]: A list of Document objects with added semantic metadata.
        """
        indexed_docs = {
            doc.id: Document(page_content=doc.page_content, metadata=doc.metadata)
            for doc in docs_with_id
        }

        for classified_doc in classified_docs:
            doc_id = classified_doc.get("id")
            if doc_id in indexed_docs:
                self._add_semantic_to_doc(indexed_docs[doc_id], classified_doc)

        semantic_metadata_docs = [doc for doc in indexed_docs.values()]

        return semantic_metadata_docs

    def _unindex_docs(self, docs_with_id: List[IndexedDocument]) -> List[Document]:
        """
        Converts a list of IndexedDocument objects to a list of Document objects.

        Args:
            docs_with_id (List[IndexedDocument]): A list of IndexedDocument objects.

        Returns:
            List[Document]: A list of Document objects.
        """
        docs = [
            Document(page_content=doc.page_content, metadata=doc.metadata)
            for i, doc in enumerate(docs_with_id)
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
