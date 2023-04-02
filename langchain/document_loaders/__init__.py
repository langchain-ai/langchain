"""All different types of document loaders."""

from langchain.document_loaders.airbyte_json import AirbyteJSONLoader
from langchain.document_loaders.apify_dataset import ApifyDatasetLoader
from langchain.document_loaders.azlyrics import AZLyricsLoader
from langchain.document_loaders.azure_blob_storage_container import (
    AzureBlobStorageContainerLoader,
)
from langchain.document_loaders.azure_blob_storage_file import (
    AzureBlobStorageFileLoader,
)
from langchain.document_loaders.bigquery import BigQueryLoader
from langchain.document_loaders.blackboard import BlackboardLoader
from langchain.document_loaders.college_confidential import CollegeConfidentialLoader
from langchain.document_loaders.conllu import CoNLLULoader
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.duckdb_loader import DuckDBLoader
from langchain.document_loaders.email import UnstructuredEmailLoader
from langchain.document_loaders.epub import UnstructuredEPubLoader
from langchain.document_loaders.evernote import EverNoteLoader
from langchain.document_loaders.facebook_chat import FacebookChatLoader
from langchain.document_loaders.gcs_directory import GCSDirectoryLoader
from langchain.document_loaders.gcs_file import GCSFileLoader
from langchain.document_loaders.gitbook import GitbookLoader
from langchain.document_loaders.googledrive import GoogleDriveLoader
from langchain.document_loaders.gutenberg import GutenbergLoader
from langchain.document_loaders.hn import HNLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.ifixit import IFixitLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders.imsdb import IMSDbLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.notebook import NotebookLoader
from langchain.document_loaders.notion import NotionDirectoryLoader
from langchain.document_loaders.notiondb import NotionDBLoader
from langchain.document_loaders.obsidian import ObsidianLoader
from langchain.document_loaders.pdf import (
    OnlinePDFLoader,
    PDFMinerLoader,
    PyMuPDFLoader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.readthedocs import ReadTheDocsLoader
from langchain.document_loaders.roam import RoamLoader
from langchain.document_loaders.s3_directory import S3DirectoryLoader
from langchain.document_loaders.s3_file import S3FileLoader
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders.srt import SRTLoader
from langchain.document_loaders.telegram import TelegramChatLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.unstructured import (
    UnstructuredFileIOLoader,
    UnstructuredFileLoader,
)
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.document_loaders.url_selenium import SeleniumURLLoader
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.document_loaders.whatsapp_chat import WhatsAppChatLoader
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
from langchain.document_loaders.youtube import (
    GoogleApiClient,
    GoogleApiYoutubeLoader,
    YoutubeLoader,
)

# Legacy: only for backwards compat. Use PyPDFLoader instead
PagedPDFSplitter = PyPDFLoader

__all__ = [
    "UnstructuredFileLoader",
    "UnstructuredFileIOLoader",
    "UnstructuredURLLoader",
    "SeleniumURLLoader",
    "DirectoryLoader",
    "NotionDirectoryLoader",
    "NotionDBLoader",
    "ReadTheDocsLoader",
    "GoogleDriveLoader",
    "UnstructuredHTMLLoader",
    "BSHTMLLoader",
    "UnstructuredPowerPointLoader",
    "UnstructuredWordDocumentLoader",
    "UnstructuredPDFLoader",
    "UnstructuredImageLoader",
    "ObsidianLoader",
    "UnstructuredEmailLoader",
    "UnstructuredEPubLoader",
    "UnstructuredMarkdownLoader",
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
    "IFixitLoader",
    "GutenbergLoader",
    "PagedPDFSplitter",
    "PyPDFLoader",
    "EverNoteLoader",
    "AirbyteJSONLoader",
    "OnlinePDFLoader",
    "PDFMinerLoader",
    "PyMuPDFLoader",
    "TelegramChatLoader",
    "SRTLoader",
    "FacebookChatLoader",
    "NotebookLoader",
    "CoNLLULoader",
    "GoogleApiYoutubeLoader",
    "GoogleApiClient",
    "CSVLoader",
    "BlackboardLoader",
    "ApifyDatasetLoader",
    "WhatsAppChatLoader",
    "DataFrameLoader",
    "AzureBlobStorageFileLoader",
    "AzureBlobStorageContainerLoader",
    "SitemapLoader",
    "DuckDBLoader",
    "BigQueryLoader",
]
