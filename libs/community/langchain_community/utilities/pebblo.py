from __future__ import annotations

import logging
import os
import pathlib
import platform
from typing import List, Optional, Tuple

from langchain_core.documents import Document
from langchain_core.env import get_runtime_environment
from langchain_core.pydantic_v1 import BaseModel

from langchain_community.document_loaders.base import BaseLoader

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "0.1.1"
CLASSIFIER_URL = os.getenv("PEBBLO_CLASSIFIER_URL", "http://localhost:8000")
PEBBLO_CLOUD_URL = os.getenv("PEBBLO_CLOUD_URL", "https://api.daxa.ai")

LOADER_DOC_URL = "/v1/loader/doc"
APP_DISCOVER_URL = "/v1/app/discover"

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
remote_db = [
    "NotionDBLoader",
    "GoogleDriveLoader",
]

LOADER_TYPE_MAPPING = {
    "file": file_loader,
    "dir": dir_loader,
    "in-memory": in_memory,
    "remote_db": remote_db,
}

SUPPORTED_LOADERS = (*file_loader, *dir_loader, *in_memory)

logger = logging.getLogger(__name__)


class IndexedDocument(Document):
    """Pebblo Indexed Document."""

    id: str
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


def index_docs(docs: List[Document]) -> List[IndexedDocument]:
    """
    Indexes the documents and returns a list of IndexedDocument objects.

    Args:
        docs (List[Document]): A list of Document objects to be indexed.

    Returns:
        List[IndexedDocument]: A list of IndexedDocument objects with unique IDs.
    """
    docs_with_id = [
        IndexedDocument(id=hex(i)[2:], **doc.dict()) for i, doc in enumerate(docs)
    ]
    return docs_with_id


def unindex_docs(docs_with_id: List[IndexedDocument]) -> List[Document]:
    """
    Converts a list of IndexedDocument objects to a list of Document objects.

    Args:
        docs_with_id (List[IndexedDocument]): A list of IndexedDocument objects.

    Returns:
        List[Document]: A list of Document objects.
    """
    docs = [
        Document(page_content=doc.page_content, metadata=doc.metadata)
        for doc in docs_with_id
    ]
    return docs
