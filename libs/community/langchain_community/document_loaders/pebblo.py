"""Pebblo's safe dataloader is a wrapper for document loaders"""

import json
import logging
import os
import uuid
from http import HTTPStatus
from typing import Any, Dict, Iterator, List, Optional

import requests
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
    ):
        if not name or not isinstance(name, str):
            raise NameError("Must specify a valid name.")
        self.app_name = name
        self.api_key = os.environ.get("PEBBLO_API_KEY") or api_key
        self.load_id = str(uuid.uuid4())
        self.loader = langchain_loader
        self.owner = owner
        self.description = description
        self.source_path = get_loader_full_path(self.loader)
        self.source_owner = PebbloSafeLoader.get_file_owner_from_path(self.source_path)
        self.docs: List[Document] = []
        loader_name = str(type(self.loader)).split(".")[-1].split("'")[0]
        self.source_type = get_loader_type(loader_name)
        self.source_path_size = self.get_source_size(self.source_path)
        self.source_aggr_size = 0
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
        self._send_loader_doc(loading_end=True)
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
                self._send_loader_doc(loading_end=True)
                break
            self.docs = [
                doc,
            ]
            self._send_loader_doc()
            yield doc

    @classmethod
    def set_discover_sent(cls) -> None:
        cls._discover_sent = True

    @classmethod
    def set_loader_sent(cls) -> None:
        cls._loader_sent = True

    def _send_loader_doc(self, loading_end: bool = False) -> list:
        """Send documents fetched from loader to pebblo-server. Then send
        classified documents to Daxa cloud(If api_key is present). Internal method.

        Args:
            loading_end (bool, optional): Flag indicating the halt of data
                                        loading by loader. Defaults to False.
        """
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        doc_content = [doc.dict() for doc in self.docs]
        docs = []
        for doc in doc_content:
            doc_authorized_identities = doc.get("metadata", {}).get(
                "authorized_identities", []
            )
            doc_source_path = get_full_path(
                doc.get("metadata", {}).get("source", self.source_path)
            )
            doc_source_owner = PebbloSafeLoader.get_file_owner_from_path(
                doc_source_path
            )
            doc_source_size = self.get_source_size(doc_source_path)
            page_content = str(doc.get("page_content"))
            page_content_size = self.calculate_content_size(page_content)
            self.source_aggr_size += page_content_size
            docs.append(
                {
                    "doc": page_content,
                    "source_path": doc_source_path,
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
        }
        if loading_end is True:
            payload["loading_end"] = "true"
            if "loader_details" in payload:
                payload["loader_details"]["source_aggr_size"] = self.source_aggr_size
        payload = Doc(**payload).dict(exclude_unset=True)
        load_doc_url = f"{CLASSIFIER_URL}{LOADER_DOC_URL}"
        classified_docs = []
        try:
            pebblo_resp = requests.post(
                load_doc_url, headers=headers, json=payload, timeout=300
            )
            classified_docs = json.loads(pebblo_resp.text).get("docs", None)
            if pebblo_resp.status_code not in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
                logger.warning(
                    "Received unexpected HTTP response code: %s",
                    pebblo_resp.status_code,
                )
            logger.debug(
                "send_loader_doc[local]: request url %s, body %s len %s\
                    response status %s body %s",
                pebblo_resp.request.url,
                str(pebblo_resp.request.body),
                str(len(pebblo_resp.request.body if pebblo_resp.request.body else [])),
                str(pebblo_resp.status_code),
                pebblo_resp.json(),
            )
        except requests.exceptions.RequestException:
            logger.warning("Unable to reach pebblo server.")
        except Exception as e:
            logger.warning("An Exception caught in _send_loader_doc: %s", e)

        if self.api_key:
            if not classified_docs:
                logger.warning("No classified docs to send to pebblo-cloud.")
                return classified_docs
            try:
                payload["docs"] = classified_docs
                payload["classified"] = True
                headers.update({"x-api-key": self.api_key})
                pebblo_cloud_url = f"{PEBBLO_CLOUD_URL}{LOADER_DOC_URL}"
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
                logger.warning("An Exception caught in _send_loader_doc: %s", e)

        if loading_end is True:
            PebbloSafeLoader.set_loader_sent()
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
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        payload = self.app.dict(exclude_unset=True)
        app_discover_url = f"{CLASSIFIER_URL}{APP_DISCOVER_URL}"
        try:
            pebblo_resp = requests.post(
                app_discover_url, headers=headers, json=payload, timeout=20
            )
            logger.debug(
                "send_discover[local]: request url %s, body %s len %s\
                    response status %s body %s",
                pebblo_resp.request.url,
                str(pebblo_resp.request.body),
                str(len(pebblo_resp.request.body if pebblo_resp.request.body else [])),
                str(pebblo_resp.status_code),
                pebblo_resp.json(),
            )
            if pebblo_resp.status_code in [HTTPStatus.OK, HTTPStatus.BAD_GATEWAY]:
                PebbloSafeLoader.set_discover_sent()
            else:
                logger.warning(
                    f"Received unexpected HTTP response code: {pebblo_resp.status_code}"
                )
        except requests.exceptions.RequestException:
            logger.warning("Unable to reach pebblo server.")
        except Exception:
            logger.warning("An Exception caught in _send_discover.")

        if self.api_key:
            try:
                headers.update({"x-api-key": self.api_key})
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
                logger.warning("An Exception caught in _send_discover: %s", e)

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
