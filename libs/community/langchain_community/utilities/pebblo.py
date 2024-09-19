from __future__ import annotations

import json
import logging
import os
import pathlib
import platform
from enum import Enum
from http import HTTPStatus
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.env import get_runtime_environment
from langchain_core.utils import get_from_dict_or_env
from pydantic import BaseModel
from requests import Response, request
from requests.exceptions import RequestException

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.1"

_DEFAULT_CLASSIFIER_URL = "http://localhost:8000"
_DEFAULT_PEBBLO_CLOUD_URL = "https://api.daxa.ai"
BATCH_SIZE_BYTES = 100 * 1024  # 100 KB

# Supported loaders for Pebblo safe data loading
file_loader = [
    "JSONLoader",
    "S3FileLoader",
    "UnstructuredMarkdownLoader",
    "UnstructuredPDFLoader",
    "UnstructuredFileLoader",
    "UnstructuredJsonLoader",
    "PyPDFLoader",
    "GCSFileLoader",
    "AmazonTextractPDFLoader",
    "CSVLoader",
    "UnstructuredExcelLoader",
    "UnstructuredEmailLoader",
]
dir_loader = [
    "DirectoryLoader",
    "S3DirLoader",
    "SlackDirectoryLoader",
    "PyPDFDirectoryLoader",
    "NotionDirectoryLoader",
]

in_memory = ["DataFrameLoader"]
cloud_folder = [
    "NotionDBLoader",
    "GoogleDriveLoader",
    "SharePointLoader",
]

LOADER_TYPE_MAPPING = {
    "file": file_loader,
    "dir": dir_loader,
    "in-memory": in_memory,
    "cloud-folder": cloud_folder,
}


class Routes(str, Enum):
    """Routes available for the Pebblo API as enumerator."""

    loader_doc = "/v1/loader/doc"
    loader_app_discover = "/v1/app/discover"


class IndexedDocument(Document):
    """Pebblo Indexed Document."""

    pb_id: str
    """Unique ID of the document."""


class Runtime(BaseModel):
    """Pebblo Runtime."""

    type: str = "local"
    """Runtime type. Defaults to 'local'."""
    host: str
    """Host name of the runtime."""
    path: str
    """Current working directory path."""
    ip: Optional[str] = ""
    """IP address of the runtime. Defaults to ''."""
    platform: str
    """Platform details of the runtime."""
    os: str
    """OS name."""
    os_version: str
    """OS version."""
    language: str
    """Runtime kernel."""
    language_version: str
    """Version of the runtime kernel."""
    runtime: str = "local"
    """More runtime details. Defaults to 'local'."""


class Framework(BaseModel):
    """Pebblo Framework instance."""

    name: str
    """Name of the Framework."""
    version: str
    """Version of the Framework."""


class App(BaseModel):
    """Pebblo AI application."""

    name: str
    """Name of the app."""
    owner: str
    """Owner of the app."""
    description: Optional[str]
    """Description of the app."""
    load_id: str
    """Unique load_id of the app instance."""
    runtime: Runtime
    """Runtime details of the app."""
    framework: Framework
    """Framework details of the app."""
    plugin_version: str
    """Plugin version used for the app."""
    client_version: Framework
    """Client version used for the app."""


class Doc(BaseModel):
    """Pebblo document."""

    name: str
    """Name of app originating this document."""
    owner: str
    """Owner of app."""
    docs: list
    """List of documents with its metadata."""
    plugin_version: str
    """Pebblo plugin Version"""
    load_id: str
    """Unique load_id of the app instance."""
    loader_details: dict
    """Loader details with its metadata."""
    loading_end: bool
    """Boolean, specifying end of loading of source."""
    source_owner: str
    """Owner of the source of the loader."""
    classifier_location: str
    """Location of the classifier."""


def get_full_path(path: str) -> str:
    """Return an absolute local path for a local file/directory,
    for a network related path, return as is.

    Args:
        path (str): Relative path to be resolved.

    Returns:
        str: Resolved absolute path.
    """
    if (
        not path
        or ("://" in path)
        or ("/" == path[0])
        or (path in ["unknown", "-", "in-memory"])
    ):
        return path
    full_path = pathlib.Path(path)
    if full_path.exists():
        full_path = full_path.resolve()
    return str(full_path)


def get_loader_type(loader: str) -> str:
    """Return loader type among, file, dir or in-memory.

    Args:
        loader (str): Name of the loader, whose type is to be resolved.

    Returns:
        str: One of the loader type among, file/dir/in-memory.
    """
    for loader_type, loaders in LOADER_TYPE_MAPPING.items():
        if loader in loaders:
            return loader_type
    return "unsupported"


def get_loader_full_path(loader: BaseLoader) -> str:
    """Return an absolute source path of source of loader based on the
    keys present in Document.

    Args:
        loader (BaseLoader): Langchain document loader, derived from Baseloader.
    """
    from langchain_community.document_loaders import (
        DataFrameLoader,
        GCSFileLoader,
        NotionDBLoader,
        S3FileLoader,
    )

    location = "-"
    if not isinstance(loader, BaseLoader):
        logger.error(
            "loader is not derived from BaseLoader, source location will be unknown!"
        )
        return location
    loader_dict = loader.__dict__
    try:
        if "bucket" in loader_dict:
            if isinstance(loader, GCSFileLoader):
                location = f"gc://{loader.bucket}/{loader.blob}"
            elif isinstance(loader, S3FileLoader):
                location = f"s3://{loader.bucket}/{loader.key}"
        elif "source" in loader_dict:
            location = loader_dict["source"]
            if location and "channel" in loader_dict:
                channel = loader_dict["channel"]
                if channel:
                    location = f"{location}/{channel}"
        elif "path" in loader_dict:
            location = loader_dict["path"]
        elif "file_path" in loader_dict:
            location = loader_dict["file_path"]
        elif "web_paths" in loader_dict:
            web_paths = loader_dict["web_paths"]
            if web_paths and isinstance(web_paths, list) and len(web_paths) > 0:
                location = web_paths[0]
        # For in-memory types:
        elif isinstance(loader, DataFrameLoader):
            location = "in-memory"
        elif isinstance(loader, NotionDBLoader):
            location = f"notiondb://{loader.database_id}"
        elif loader.__class__.__name__ == "GoogleDriveLoader":
            if loader_dict.get("folder_id"):
                folder_id = loader_dict.get("folder_id")
                location = f"https://drive.google.com/drive/u/2/folders/{folder_id}"
            elif loader_dict.get("file_ids"):
                file_ids = loader_dict.get("file_ids", [])
                location = ", ".join(
                    [
                        f"https://drive.google.com/file/d/{file_id}/view"
                        for file_id in file_ids
                    ]
                )
            elif loader_dict.get("document_ids"):
                document_ids = loader_dict.get("document_ids", [])
                location = ", ".join(
                    [
                        f"https://docs.google.com/document/d/{doc_id}/edit"
                        for doc_id in document_ids
                    ]
                )

    except Exception:
        pass
    return get_full_path(str(location))


def get_runtime() -> Tuple[Framework, Runtime]:
    """Fetch the current Framework and Runtime details.

    Returns:
        Tuple[Framework, Runtime]: Framework and Runtime for the current app instance.
    """
    runtime_env = get_runtime_environment()
    framework = Framework(
        name="langchain", version=runtime_env.get("library_version", None)
    )
    uname = platform.uname()
    runtime = Runtime(
        host=uname.node,
        path=os.environ["PWD"],
        platform=runtime_env.get("platform", "unknown"),
        os=uname.system,
        os_version=uname.version,
        ip=get_ip(),
        language=runtime_env.get("runtime", "unknown"),
        language_version=runtime_env.get("runtime_version", "unknown"),
    )

    if "Darwin" in runtime.os:
        runtime.type = "desktop"
        runtime.runtime = "Mac OSX"

    logger.debug(f"framework {framework}")
    logger.debug(f"runtime {runtime}")
    return framework, runtime


def get_ip() -> str:
    """Fetch local runtime ip address.

    Returns:
        str: IP address
    """
    import socket  # lazy imports

    host = socket.gethostname()
    try:
        public_ip = socket.gethostbyname(host)
    except Exception:
        public_ip = socket.gethostbyname("localhost")
    return public_ip


def generate_size_based_batches(
    docs: List[Document], max_batch_size: int = 100 * 1024
) -> List[List[Document]]:
    """
    Generate batches of documents based on page_content size.
    Args:
        docs: List of documents to be batched.
        max_batch_size: Maximum size of each batch in bytes. Defaults to 100*1024(100KB)
    Returns:
        List[List[Document]]: List of batches of documents
    """
    batches: List[List[Document]] = []
    current_batch: List[Document] = []
    current_batch_size: int = 0

    for doc in docs:
        # Calculate the size of the document in bytes
        doc_size: int = len(doc.page_content.encode("utf-8"))

        if doc_size > max_batch_size:
            # If a single document exceeds the max batch size, send it as a single batch
            batches.append([doc])
        else:
            if current_batch_size + doc_size > max_batch_size:
                # If adding this document exceeds the max batch size, start a new batch
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0

            # Add document to the current batch
            current_batch.append(doc)
            current_batch_size += doc_size

    # Add the last batch if it has documents
    if current_batch:
        batches.append(current_batch)

    return batches


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


def get_source_size(source_path: str) -> int:
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


def calculate_content_size(data: str) -> int:
    """Calculate the content size in bytes:
    - Encode the string to bytes using a specific encoding (e.g., UTF-8)
    - Get the length of the encoded bytes.

    Args:
        data (str): Data string.

    Returns:
        int: Size of string in bytes.
    """
    encoded_content = data.encode("utf-8")
    size = len(encoded_content)
    return size


class PebbloLoaderAPIWrapper(BaseModel):
    """Wrapper for Pebblo Loader API."""

    api_key: Optional[str]  # Use SecretStr
    """API key for Pebblo Cloud"""
    classifier_location: str = "local"
    """Location of the classifier, local or cloud. Defaults to 'local'"""
    classifier_url: Optional[str]
    """URL of the Pebblo Classifier"""
    cloud_url: Optional[str]
    """URL of the Pebblo Cloud"""

    def __init__(self, **kwargs: Any):
        """Validate that api key in environment."""
        kwargs["api_key"] = get_from_dict_or_env(
            kwargs, "api_key", "PEBBLO_API_KEY", ""
        )
        kwargs["classifier_url"] = get_from_dict_or_env(
            kwargs, "classifier_url", "PEBBLO_CLASSIFIER_URL", _DEFAULT_CLASSIFIER_URL
        )
        kwargs["cloud_url"] = get_from_dict_or_env(
            kwargs, "cloud_url", "PEBBLO_CLOUD_URL", _DEFAULT_PEBBLO_CLOUD_URL
        )
        super().__init__(**kwargs)

    def send_loader_discover(self, app: App) -> None:
        """
        Send app discovery request to Pebblo server & cloud.

        Args:
            app (App): App instance to be discovered.
        """
        pebblo_resp = None
        payload = app.dict(exclude_unset=True)

        if self.classifier_location == "local":
            # Send app details to local classifier
            headers = self._make_headers()
            app_discover_url = (
                f"{self.classifier_url}{Routes.loader_app_discover.value}"
            )
            pebblo_resp = self.make_request("POST", app_discover_url, headers, payload)

        if self.api_key:
            # Send app details to Pebblo cloud if api_key is present
            headers = self._make_headers(cloud_request=True)
            if pebblo_resp:
                pebblo_server_version = json.loads(pebblo_resp.text).get(
                    "pebblo_server_version"
                )
                payload.update({"pebblo_server_version": pebblo_server_version})

            payload.update({"pebblo_client_version": PLUGIN_VERSION})
            pebblo_cloud_url = f"{self.cloud_url}{Routes.loader_app_discover.value}"
            _ = self.make_request("POST", pebblo_cloud_url, headers, payload)

    def classify_documents(
        self,
        docs_with_id: List[IndexedDocument],
        app: App,
        loader_details: dict,
        loading_end: bool = False,
    ) -> dict:
        """
        Send documents to Pebblo server for classification.
        Then send classified documents to Daxa cloud(If api_key is present).

        Args:
            docs_with_id (List[IndexedDocument]): List of documents to be classified.
            app (App): App instance.
            loader_details (dict): Loader details.
            loading_end (bool): Boolean, indicating the halt of data loading by loader.
        """
        source_path = loader_details.get("source_path", "")
        source_owner = get_file_owner_from_path(source_path)
        # Prepare docs for classification
        docs, source_aggregate_size = self.prepare_docs_for_classification(
            docs_with_id, source_path, loader_details
        )
        # Build payload for classification
        payload = self.build_classification_payload(
            app, docs, loader_details, source_owner, source_aggregate_size, loading_end
        )

        classified_docs = {}
        if self.classifier_location == "local":
            # Send docs to local classifier
            headers = self._make_headers()
            load_doc_url = f"{self.classifier_url}{Routes.loader_doc.value}"
            try:
                pebblo_resp = self.make_request(
                    "POST", load_doc_url, headers, payload, 300
                )

                if pebblo_resp:
                    # Updating structure of pebblo response docs for efficient searching
                    for classified_doc in json.loads(pebblo_resp.text).get("docs", []):
                        classified_docs.update(
                            {classified_doc["pb_id"]: classified_doc}
                        )
            except Exception as e:
                logger.warning("An Exception caught in classify_documents: local %s", e)

        if self.api_key:
            # Send docs to Pebblo cloud if api_key is present
            if self.classifier_location == "local":
                # If local classifier is used add the classified information
                # and remove doc content
                self.update_doc_data(payload["docs"], classified_docs)
            self.send_docs_to_pebblo_cloud(payload)
        elif self.classifier_location == "pebblo-cloud":
            logger.warning("API key is missing for sending docs to Pebblo cloud.")
            raise NameError("API key is missing for sending docs to Pebblo cloud.")

        return classified_docs

    def send_docs_to_pebblo_cloud(self, payload: dict) -> None:
        """
        Send documents to Pebblo cloud.

        Args:
            payload (dict): The payload containing documents to be sent.
        """
        headers = self._make_headers(cloud_request=True)
        pebblo_cloud_url = f"{self.cloud_url}{Routes.loader_doc.value}"
        try:
            _ = self.make_request("POST", pebblo_cloud_url, headers, payload)
        except Exception as e:
            logger.warning("An Exception caught in classify_documents: cloud %s", e)

    def _make_headers(self, cloud_request: bool = False) -> dict:
        """
        Generate headers for the request.

        args:
            cloud_request (bool): flag indicating whether the request is for Pebblo
            cloud.
        returns:
            dict: Headers for the request.

        """
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if cloud_request:
            # Add API key for Pebblo cloud request
            if self.api_key:
                headers.update({"x-api-key": self.api_key})
            else:
                logger.warning("API key is missing for Pebblo cloud request.")
        return headers

    def build_classification_payload(
        self,
        app: App,
        docs: List[dict],
        loader_details: dict,
        source_owner: str,
        source_aggregate_size: int,
        loading_end: bool,
    ) -> dict:
        """
        Build the payload for document classification.

        Args:
            app (App): App instance.
            docs (List[dict]): List of documents to be classified.
            loader_details (dict): Loader details.
            source_owner (str): Owner of the source.
            source_aggregate_size (int): Aggregate size of the source.
            loading_end (bool): Boolean indicating the halt of data loading by loader.

        Returns:
            dict: Payload for document classification.
        """
        payload: Dict[str, Any] = {
            "name": app.name,
            "owner": app.owner,
            "docs": docs,
            "plugin_version": PLUGIN_VERSION,
            "load_id": app.load_id,
            "loader_details": loader_details,
            "loading_end": "false",
            "source_owner": source_owner,
            "classifier_location": self.classifier_location,
        }
        if loading_end is True:
            payload["loading_end"] = "true"
            if "loader_details" in payload:
                payload["loader_details"]["source_aggregate_size"] = (
                    source_aggregate_size
                )
        payload = Doc(**payload).dict(exclude_unset=True)
        return payload

    @staticmethod
    def make_request(
        method: str,
        url: str,
        headers: dict,
        payload: Optional[dict] = None,
        timeout: int = 20,
    ) -> Optional[Response]:
        """
        Make a request to the Pebblo API

        Args:
            method (str): HTTP method (GET, POST, PUT, DELETE, etc.).
            url (str): URL for the request.
            headers (dict): Headers for the request.
            payload (Optional[dict]): Payload for the request (for POST, PUT, etc.).
            timeout (int): Timeout for the request in seconds.

        Returns:
            Optional[Response]: Response object if the request is successful.
        """
        try:
            response = request(
                method=method, url=url, headers=headers, json=payload, timeout=timeout
            )
            logger.debug(
                "Request: method %s, url %s, len %s response status %s",
                method,
                response.request.url,
                str(len(response.request.body if response.request.body else [])),
                str(response.status_code),
            )

            if response.status_code >= HTTPStatus.INTERNAL_SERVER_ERROR:
                logger.warning(f"Pebblo Server: Error {response.status_code}")
            elif response.status_code >= HTTPStatus.BAD_REQUEST:
                logger.warning(f"Pebblo received an invalid payload: {response.text}")
            elif response.status_code != HTTPStatus.OK:
                logger.warning(
                    f"Pebblo returned an unexpected response code: "
                    f"{response.status_code}"
                )

            return response
        except RequestException:
            logger.warning("Unable to reach server %s", url)
        except Exception as e:
            logger.warning("An Exception caught in make_request: %s", e)
        return None

    @staticmethod
    def prepare_docs_for_classification(
        docs_with_id: List[IndexedDocument],
        source_path: str,
        loader_details: dict,
    ) -> Tuple[List[dict], int]:
        """
        Prepare documents for classification.

        Args:
            docs_with_id (List[IndexedDocument]): List of documents to be classified.
            source_path (str): Source path of the documents.
            loader_details (dict): Contains loader info.

        Returns:
            Tuple[List[dict], int]: Documents and the aggregate size
            of the source.
        """
        docs = []
        source_aggregate_size = 0
        doc_content = [doc.dict() for doc in docs_with_id]
        source_path_update = False
        for doc in doc_content:
            doc_metadata = doc.get("metadata", {})
            doc_authorized_identities = doc_metadata.get("authorized_identities", [])
            if loader_details["loader"] == "SharePointLoader":
                doc_source_path = get_full_path(
                    doc_metadata.get("source", loader_details["source_path"])
                )
            else:
                doc_source_path = get_full_path(
                    doc_metadata.get(
                        "full_path",
                        doc_metadata.get("source", source_path),
                    )
                )
            doc_source_owner = doc_metadata.get(
                "owner", get_file_owner_from_path(doc_source_path)
            )
            doc_source_size = doc_metadata.get("size", get_source_size(doc_source_path))
            page_content = str(doc.get("page_content"))
            page_content_size = calculate_content_size(page_content)
            source_aggregate_size += page_content_size
            doc_id = doc.get("pb_id", None) or 0
            docs.append(
                {
                    "doc": page_content,
                    "source_path": doc_source_path,
                    "pb_id": doc_id,
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
            if (
                loader_details["loader"] == "SharePointLoader"
                and not source_path_update
            ):
                loader_details["source_path"] = doc_metadata.get("source_full_url")
                source_path_update = True
        return docs, source_aggregate_size

    @staticmethod
    def update_doc_data(docs: List[dict], classified_docs: dict) -> None:
        """
        Update the document data with classified information.

        Args:
            docs (List[dict]): List of document data to be updated.
            classified_docs (dict): The dictionary containing classified documents.
        """
        for doc_data in docs:
            classified_data = classified_docs.get(doc_data["pb_id"], {})
            # Update the document data with classified information
            doc_data.update(
                {
                    "pb_checksum": classified_data.get("pb_checksum"),
                    "loader_source_path": classified_data.get("loader_source_path"),
                    "entities": classified_data.get("entities", {}),
                    "topics": classified_data.get("topics", {}),
                }
            )
            # Remove the document content
            doc_data.pop("doc")
