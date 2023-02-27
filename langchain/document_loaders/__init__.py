"""All different types of document loaders."""

from langchain.document_loaders.airbyte_json import AirbyteJSONLoader
from langchain.document_loaders.azlyrics import AZLyricsLoader
from langchain.document_loaders.college_confidential import CollegeConfidentialLoader
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.docx import UnstructuredDocxLoader
from langchain.document_loaders.email import UnstructuredEmailLoader
from langchain.document_loaders.evernote import EverNoteLoader
from langchain.document_loaders.facebook_chat import FacebookChatLoader
from langchain.document_loaders.gcs_directory import GCSDirectoryLoader
from langchain.document_loaders.gcs_file import GCSFileLoader
from langchain.document_loaders.gitbook import GitbookLoader
from langchain.document_loaders.googledrive import GoogleDriveLoader
from langchain.document_loaders.gutenberg import GutenbergLoader
from langchain.document_loaders.hn import HNLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.document_loaders.imsdb import IMSDbLoader
from langchain.document_loaders.notebook import NotebookLoader
from langchain.document_loaders.notion import NotionDirectoryLoader
from langchain.document_loaders.obsidian import ObsidianLoader
from langchain.document_loaders.online_pdf import OnlinePDFLoader
from langchain.document_loaders.paged_pdf import PagedPDFSplitter
from langchain.document_loaders.pdf import PDFMinerLoader, UnstructuredPDFLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.readthedocs import ReadTheDocsLoader
from langchain.document_loaders.roam import RoamLoader
from langchain.document_loaders.s3_directory import S3DirectoryLoader
from langchain.document_loaders.s3_file import S3FileLoader
from langchain.document_loaders.srt import SRTLoader
from langchain.document_loaders.telegram import TelegramChatLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import (
    UnstructuredFileIOLoader,
    UnstructuredFileLoader,
)
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.document_loaders.youtube import YoutubeLoader

__all__ = [
    "UnstructuredFileLoader",
    "UnstructuredFileIOLoader",
    "UnstructuredURLLoader",
    "DirectoryLoader",
    "NotionDirectoryLoader",
    "ReadTheDocsLoader",
    "GoogleDriveLoader",
    "UnstructuredHTMLLoader",
    "UnstructuredPowerPointLoader",
    "UnstructuredWordDocumentLoader",
    "UnstructuredPDFLoader",
    "ObsidianLoader",
    "UnstructuredDocxLoader",
    "UnstructuredEmailLoader",
    "RoamLoader",
    "YoutubeLoader",
    "S3FileLoader",
    "TextLoader",
    "HNLoader",
    "GitbookLoader",
    "S3DirectoryLoader",
    "GCSFileLoader",
    "GCSDirectoryLoader",
    "WebBaseLoader",
    "IMSDbLoader",
    "AZLyricsLoader",
    "CollegeConfidentialLoader",
    "GutenbergLoader",
    "PagedPDFSplitter",
    "EverNoteLoader",
    "AirbyteJSONLoader",
    "OnlinePDFLoader",
    "PDFMinerLoader",
    "TelegramChatLoader",
    "SRTLoader",
    "FacebookChatLoader",
    "NotebookLoader",
]
