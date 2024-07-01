"""Pebblo's safe dataloader is a wrapper for document loaders"""

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
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
    calculate_content_size,
    get_file_owner_from_path,
    get_full_path,
    get_loader_full_path,
    get_loader_type,
    get_runtime,
    get_source_size,
    index_docs,
    unindex_docs,
)

logger = logging.getLogger(__name__)


class PebbloBaseLoader(BaseLoader, ABC):
    """Pebblo Base Loader is a Base class to be inherited by all Pebblo loaders."""

    _discover_sent_local: bool = False
    _loader_sent_local: bool = False
    _discover_sent_cloud: bool = False
    _loader_sent_cloud: bool = False

    def __init__(
        self,
        langchain_loader: BaseLoader,
        name: str,
        owner: str = "",
        description: str = "",
        api_key: Optional[str] = None,
        classifier_url: Optional[str] = None,
        *,
        classifier_location: str = "local",
    ):
        """
        Base class of all Pebblo Loaders.

        Args:
            langchain_loader (BaseLoader): The base loader for the
                Pebblo document loader.
            name (str): The name of the app.
            owner (str, optional): The owner of the app.
                Defaults to an empty string.
            description (str, optional): The description of the app.
                Defaults to an empty string.
            api_key (str, optional): The API key for the app.
                Defaults to None.
            classifier_url (str, optional): The URL for the classifier.
                Defaults to None.
            classifier_location (str, optional): The location of the classifier.
                Defaults to "local".
        """
        if not name or not isinstance(name, str):
            raise NameError("Must specify a valid app name.")
        self.app_name = name
        self.api_key = os.environ.get("PEBBLO_API_KEY") or api_key
        self.load_id = str(uuid.uuid4())
        self.loader = langchain_loader
        self.owner = owner
        self.description = description
        self.docs: List[Document] = []
        self.docs_with_id: Union[List[IndexedDocument], List[Document], List] = []
        self.source_aggregate_size = 0
        self.classifier_url = classifier_url or CLASSIFIER_URL
        self.classifier_location = classifier_location

    @classmethod
    def set_discover_local(cls) -> None:
        cls._discover_sent_local = True

    @classmethod
    def set_loader_local(cls) -> None:
        cls._loader_sent_local = True

    @classmethod
    def set_discover_cloud(cls) -> None:
        cls._discover_sent_cloud = True

    @classmethod
    def set_loader_cloud(cls) -> None:
        cls._loader_sent_local = True

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

    @abstractmethod
    def load(self) -> List[Document]:
        """Load Documents.

        Raises:
            NotImplementedError: raised when lazy_load is not implemented
            within wrapped loader.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()."
        )

    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        """Load documents in lazy fashion.

        Raises:
            NotImplementedError: raised when lazy_load is not implemented
            within wrapped loader.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement lazy_load()."
        )


class PebbloSafeLoader(PebbloBaseLoader):
    """Pebblo Safe Loader class is a wrapper around document loaders enabling the data
    to be scrutinized.
    """

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
        """
        Initialize a Pebblo document loader.

        Args:
            langchain_loader (BaseLoader): The base loader instance.
            name (str): The name of the document loader.
            owner (str, optional): The owner of the document loader. Defaults to "".
            description (str, optional): The description of the document loader.
                Defaults to "".
            api_key (str, optional): The API key for the document loader.
                Defaults to None.
            load_semantic (bool, optional): Whether to enable semantic information
                to metadata of the Document. Defaults to False.
            classifier_url (str, optional): The URL of the classifier.
                Defaults to None.
            classifier_location (str, optional): The location of the classifier.
                Defaults to "local".
        """
        if not name or not isinstance(name, str):
            raise NameError("Must specify a valid name.")
        super().__init__(
            langchain_loader=langchain_loader,
            name=name,
            owner=owner,
            description=description,
            api_key=api_key,
            classifier_url=classifier_url,
            classifier_location=classifier_location,
        )
        self.load_semantic = os.environ.get("PEBBLO_LOAD_SEMANTIC") or load_semantic
        self._init_loader_details()
        # generate app
        self.app = self._get_app_details()
        self._send_discover()

    def load(self) -> List[Document]:
        """Load Documents.

        Returns:
            list: Documents fetched from load method of the wrapped `loader`.
        """
        self.docs = self.loader.load()
        if not self.docs:
            logger.debug("No documents found in loader.")
            return self.docs
        if not self.load_semantic:
            self._classify_doc(self.docs, loading_end=True)
            return self.docs
        docs_with_id = index_docs(self.docs)
        classified_docs = self._classify_doc(docs_with_id, loading_end=True)
        docs_with_semantics = self._add_semantic_to_docs(docs_with_id, classified_docs)
        self.docs = unindex_docs(docs_with_semantics)  # type: ignore
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
                self.docs_with_id = index_docs(self.docs)
                classified_doc = self._classify_doc(self.docs)
                docs_with_semantics = self._add_semantic_to_docs(
                    self.docs_with_id, classified_doc
                )
                self.docs = unindex_docs(docs_with_semantics)  # type: ignore
                yield self.docs[0]

    def _init_loader_details(self) -> None:
        """Initialize loader details."""
        self.source_path = get_loader_full_path(self.loader)
        self.source_owner = get_file_owner_from_path(self.source_path)
        loader_name = str(type(self.loader)).split(".")[-1].split("'")[0]
        self.source_type = get_loader_type(loader_name)
        self.source_path_size = get_source_size(self.source_path)
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
        classified_docs = []
        payload = self._get_loader_doc_payload(loaded_docs, loading_end=loading_end)
        payload = Doc(**payload).dict(exclude_unset=True)
        # Raw payload to be sent to classifier
        if self.classifier_location == "local":
            load_doc_url = f"{self.classifier_url}{LOADER_DOC_URL}"
            pebblo_resp = self._request_pebblo(
                load_doc_url, headers, payload, timeout=300
            )
            if pebblo_resp is not None:
                classified_docs = json.loads(pebblo_resp.text).get("docs", None)
                self.set_loader_local()

        if self.api_key:
            if self.classifier_location == "local":
                payload["docs"] = classified_docs
            headers.update({"x-api-key": self.api_key})
            pebblo_cloud_url = f"{PEBBLO_CLOUD_URL}{LOADER_DOC_URL}"
            pebblo_resp = self._request_pebblo(pebblo_cloud_url, headers, payload)
            if pebblo_resp is not None:
                classified_docs = json.loads(pebblo_resp.text).get("docs", None)
                self.set_loader_cloud()
        elif self.classifier_location == "pebblo-cloud":
            logger.warning("API key is missing for sending docs to Pebblo cloud.")
            raise NameError("API key is missing for sending docs to Pebblo cloud.")

        return classified_docs if self.load_semantic else []

    def _get_loader_doc_payload(self, loaded_docs: list, loading_end: bool) -> dict:
        """Get the payload to be sent to the classifier. Internal method.

        Args:
            loaded_docs (list): List of documents fetched from loader's load operation.

        Returns:
            dict: The payload to be sent to the classifier.
        """
        doc_content = [doc.dict() for doc in loaded_docs]
        docs = []
        for doc in doc_content:
            doc_metadata = doc.get("metadata", {})
            doc_authorized_identities = doc_metadata.get("authorized_identities", [])
            doc_source_path = get_full_path(
                doc_metadata.get(
                    "full_path", doc_metadata.get("source", self.source_path)
                )
            )
            doc_source_owner = doc_metadata.get(
                "owner", get_file_owner_from_path(doc_source_path)
            )
            doc_source_size = doc_metadata.get("size", get_source_size(doc_source_path))
            page_content = str(doc.get("page_content"))
            page_content_size = calculate_content_size(page_content)
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
        return payload

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
            pebblo_resp = self._request_pebblo(app_discover_url, headers, payload)
            if pebblo_resp is not None:
                self.set_discover_local()

        if self.api_key:
            headers.update({"x-api-key": self.api_key})
            # If the pebblo_resp is None,
            # then the pebblo server version is not available
            if pebblo_resp is not None:
                pebblo_server_version = json.loads(pebblo_resp.text).get(
                    "pebblo_server_version"
                )
                payload.update({"pebblo_server_version": pebblo_server_version})
            payload.update({"pebblo_client_version": PLUGIN_VERSION})
            pebblo_cloud_url = f"{PEBBLO_CLOUD_URL}{APP_DISCOVER_URL}"
            sent_dis_cloud = self._request_pebblo(pebblo_cloud_url, headers, payload)
            if sent_dis_cloud:
                self.set_discover_cloud()

    def _request_pebblo(
        self,
        url: str,
        headers: dict,
        payload: dict,
        timeout: int = 20,
    ) -> Optional[requests.Response]:
        """Send request to Pebblo server. Internal method.

        Args:
            url (str): The URL of the Pebblo server.
            headers (dict): The headers to be included in the request.
            payload (dict): The payload to be sent in the request body.
            timeout (int, optional): The timeout for the request. Defaults to 20.

        Returns:
            bool: True if the request was successful (status code 200 or 502),
            False otherwise.
        """
        if url.endswith(APP_DISCOVER_URL):
            stage = "send_discover"
        elif url.endswith(LOADER_DOC_URL):
            stage = "send_loader_doc"
        else:
            stage = "unset api stage"

        if url.startswith(CLASSIFIER_URL):
            location = "local"
        elif url.startswith(PEBBLO_CLOUD_URL):
            location = "cloud"
        else:
            location = "unset location"

        pebblo_resp = None
        try:
            pebblo_resp = requests.post(
                url, headers=headers, json=payload, timeout=timeout
            )
            logger.debug(
                "%s[%s]: request url %s, body %s len %s\
                    response status %s body %s",
                stage,
                location,
                pebblo_resp.request.url,
                str(pebblo_resp.request.body),
                str(len(pebblo_resp.request.body if pebblo_resp.request.body else [])),
                str(pebblo_resp.status_code),
                pebblo_resp.json(),
            )
            if pebblo_resp.status_code in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
                return pebblo_resp
            else:
                logger.warning(
                    "Received unexpected HTTP response code: %s",
                    pebblo_resp.status_code,
                )
                return None
        except requests.exceptions.RequestException:
            logger.warning("Unable to reach %s server.", location)
        except Exception as e:
            logger.warning("An Exception caught in %s[%s]: %s", stage, location, e)
        return pebblo_resp

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
