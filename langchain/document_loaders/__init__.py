"""All different types of document loaders."""

from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.docx import UnstructuredDocxLoader
from langchain.document_loaders.email import UnstructuredEmailLoader
from langchain.document_loaders.googledrive import GoogleDriveLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.document_loaders.notion import NotionDirectoryLoader
from langchain.document_loaders.obsidian import ObsidianLoader
from langchain.document_loaders.pdf import UnstructuredPDFLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.readthedocs import ReadTheDocsLoader
from langchain.document_loaders.unstructured import UnstructuredFileLoader

__all__ = [
    "UnstructuredFileLoader",
    "DirectoryLoader",
    "NotionDirectoryLoader",
    "ReadTheDocsLoader",
    "GoogleDriveLoader",
    "UnstructuredHTMLLoader",
    "UnstructuredPowerPointLoader",
    "UnstructuredPDFLoader",
    "ObsidianLoader",
    "UnstructuredDocxLoader",
    "UnstructuredEmailLoader",
]
