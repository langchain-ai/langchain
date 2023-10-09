"""**Document Loaders**  are classes to load Documents.

**Document Loaders** are usually used to load a lot of Documents in a single run.

**Class hierarchy:**

.. code-block::

    BaseLoader --> <name>Loader  # Examples: TextLoader, UnstructuredFileLoader

**Main helpers:**

.. code-block::

    Document, <name>TextSplitter
"""

from langchain.document_loaders.acreom import AcreomLoader
from langchain.document_loaders.airbyte import (
    AirbyteCDKLoader,
    AirbyteGongLoader,
    AirbyteHubspotLoader,
    AirbyteSalesforceLoader,
    AirbyteShopifyLoader,
    AirbyteStripeLoader,
    AirbyteTypeformLoader,
    AirbyteZendeskSupportLoader,
)
from langchain.document_loaders.airbyte_json import AirbyteJSONLoader
from langchain.document_loaders.airtable import AirtableLoader
from langchain.document_loaders.apify_dataset import ApifyDatasetLoader
from langchain.document_loaders.arcgis_loader import ArcGISLoader
from langchain.document_loaders.arxiv import ArxivLoader
from langchain.document_loaders.assemblyai import AssemblyAIAudioTranscriptLoader
from langchain.document_loaders.async_html import AsyncHtmlLoader
from langchain.document_loaders.azlyrics import AZLyricsLoader
from langchain.document_loaders.azure_blob_storage_container import (
    AzureBlobStorageContainerLoader,
)
from langchain.document_loaders.azure_blob_storage_file import (
    AzureBlobStorageFileLoader,
)
from langchain.document_loaders.bibtex import BibtexLoader
from langchain.document_loaders.bigquery import BigQueryLoader
from langchain.document_loaders.bilibili import BiliBiliLoader
from langchain.document_loaders.blackboard import BlackboardLoader
from langchain.document_loaders.blob_loaders import (
    Blob,
    BlobLoader,
    FileSystemBlobLoader,
    YoutubeAudioLoader,
)
from langchain.document_loaders.blockchain import BlockchainDocumentLoader
from langchain.document_loaders.brave_search import BraveSearchLoader
from langchain.document_loaders.browserless import BrowserlessLoader
from langchain.document_loaders.chatgpt import ChatGPTLoader
from langchain.document_loaders.chromium import AsyncChromiumLoader
from langchain.document_loaders.college_confidential import CollegeConfidentialLoader
from langchain.document_loaders.concurrent import ConcurrentLoader
from langchain.document_loaders.confluence import ConfluenceLoader
from langchain.document_loaders.conllu import CoNLLULoader
from langchain.document_loaders.csv_loader import CSVLoader, UnstructuredCSVLoader
from langchain.document_loaders.cube_semantic import CubeSemanticLoader
from langchain.document_loaders.datadog_logs import DatadogLogsLoader
from langchain.document_loaders.dataframe import DataFrameLoader
from langchain.document_loaders.diffbot import DiffbotLoader
from langchain.document_loaders.directory import DirectoryLoader
from langchain.document_loaders.discord import DiscordChatLoader
from langchain.document_loaders.docugami import DocugamiLoader
from langchain.document_loaders.dropbox import DropboxLoader
from langchain.document_loaders.duckdb_loader import DuckDBLoader
from langchain.document_loaders.email import (
    OutlookMessageLoader,
    UnstructuredEmailLoader,
)
from langchain.document_loaders.embaas import EmbaasBlobLoader, EmbaasLoader
from langchain.document_loaders.epub import UnstructuredEPubLoader
from langchain.document_loaders.etherscan import EtherscanLoader
from langchain.document_loaders.evernote import EverNoteLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.facebook_chat import FacebookChatLoader
from langchain.document_loaders.fauna import FaunaLoader
from langchain.document_loaders.figma import FigmaFileLoader
from langchain.document_loaders.gcs_directory import GCSDirectoryLoader
from langchain.document_loaders.gcs_file import GCSFileLoader
from langchain.document_loaders.geodataframe import GeoDataFrameLoader
from langchain.document_loaders.git import GitLoader
from langchain.document_loaders.gitbook import GitbookLoader
from langchain.document_loaders.github import GitHubIssuesLoader
from langchain.document_loaders.googledrive import GoogleDriveLoader
from langchain.document_loaders.gutenberg import GutenbergLoader
from langchain.document_loaders.hn import HNLoader
from langchain.document_loaders.html import UnstructuredHTMLLoader
from langchain.document_loaders.html_bs import BSHTMLLoader
from langchain.document_loaders.hugging_face_dataset import HuggingFaceDatasetLoader
from langchain.document_loaders.ifixit import IFixitLoader
from langchain.document_loaders.image import UnstructuredImageLoader
from langchain.document_loaders.image_captions import ImageCaptionLoader
from langchain.document_loaders.imsdb import IMSDbLoader
from langchain.document_loaders.iugu import IuguLoader
from langchain.document_loaders.joplin import JoplinLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain.document_loaders.larksuite import LarkSuiteDocLoader
from langchain.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain.document_loaders.mastodon import MastodonTootsLoader
from langchain.document_loaders.max_compute import MaxComputeLoader
from langchain.document_loaders.mediawikidump import MWDumpLoader
from langchain.document_loaders.merge import MergedDataLoader
from langchain.document_loaders.mhtml import MHTMLLoader
from langchain.document_loaders.modern_treasury import ModernTreasuryLoader
from langchain.document_loaders.mongodb import MongodbLoader
from langchain.document_loaders.news import NewsURLLoader
from langchain.document_loaders.notebook import NotebookLoader
from langchain.document_loaders.notion import NotionDirectoryLoader
from langchain.document_loaders.notiondb import NotionDBLoader
from langchain.document_loaders.obs_directory import OBSDirectoryLoader
from langchain.document_loaders.obs_file import OBSFileLoader
from langchain.document_loaders.obsidian import ObsidianLoader
from langchain.document_loaders.odt import UnstructuredODTLoader
from langchain.document_loaders.onedrive import OneDriveLoader
from langchain.document_loaders.onedrive_file import OneDriveFileLoader
from langchain.document_loaders.open_city_data import OpenCityDataLoader
from langchain.document_loaders.org_mode import UnstructuredOrgModeLoader
from langchain.document_loaders.pdf import (
    AmazonTextractPDFLoader,
    MathpixPDFLoader,
    OnlinePDFLoader,
    PDFMinerLoader,
    PDFMinerPDFasHTMLLoader,
    PDFPlumberLoader,
    PyMuPDFLoader,
    PyPDFDirectoryLoader,
    PyPDFium2Loader,
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from langchain.document_loaders.polars_dataframe import PolarsDataFrameLoader
from langchain.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain.document_loaders.psychic import PsychicLoader
from langchain.document_loaders.pubmed import PubMedLoader
from langchain.document_loaders.pyspark_dataframe import PySparkDataFrameLoader
from langchain.document_loaders.python import PythonLoader
from langchain.document_loaders.readthedocs import ReadTheDocsLoader
from langchain.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.document_loaders.reddit import RedditPostsLoader
from langchain.document_loaders.roam import RoamLoader
from langchain.document_loaders.rocksetdb import RocksetLoader
from langchain.document_loaders.rss import RSSFeedLoader
from langchain.document_loaders.rst import UnstructuredRSTLoader
from langchain.document_loaders.rtf import UnstructuredRTFLoader
from langchain.document_loaders.s3_directory import S3DirectoryLoader
from langchain.document_loaders.s3_file import S3FileLoader
from langchain.document_loaders.sharepoint import SharePointLoader
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders.slack_directory import SlackDirectoryLoader
from langchain.document_loaders.snowflake_loader import SnowflakeLoader
from langchain.document_loaders.spreedly import SpreedlyLoader
from langchain.document_loaders.srt import SRTLoader
from langchain.document_loaders.stripe import StripeLoader
from langchain.document_loaders.telegram import (
    TelegramChatApiLoader,
    TelegramChatFileLoader,
)
from langchain.document_loaders.tencent_cos_directory import TencentCOSDirectoryLoader
from langchain.document_loaders.tencent_cos_file import TencentCOSFileLoader
from langchain.document_loaders.tensorflow_datasets import TensorflowDatasetLoader
from langchain.document_loaders.text import TextLoader
from langchain.document_loaders.tomarkdown import ToMarkdownLoader
from langchain.document_loaders.toml import TomlLoader
from langchain.document_loaders.trello import TrelloLoader
from langchain.document_loaders.tsv import UnstructuredTSVLoader
from langchain.document_loaders.twitter import TwitterTweetLoader
from langchain.document_loaders.unstructured import (
    UnstructuredAPIFileIOLoader,
    UnstructuredAPIFileLoader,
    UnstructuredFileIOLoader,
    UnstructuredFileLoader,
)
from langchain.document_loaders.url import UnstructuredURLLoader
from langchain.document_loaders.url_playwright import PlaywrightURLLoader
from langchain.document_loaders.url_selenium import SeleniumURLLoader
from langchain.document_loaders.weather import WeatherDataLoader
from langchain.document_loaders.web_base import WebBaseLoader
from langchain.document_loaders.whatsapp_chat import WhatsAppChatLoader
from langchain.document_loaders.wikipedia import WikipediaLoader
from langchain.document_loaders.word_document import (
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
)
from langchain.document_loaders.xml import UnstructuredXMLLoader
from langchain.document_loaders.xorbits import XorbitsLoader
from langchain.document_loaders.youtube import (
    GoogleApiClient,
    GoogleApiYoutubeLoader,
    YoutubeLoader,
)

# Legacy: only for backwards compatibility. Use PyPDFLoader instead
PagedPDFSplitter = PyPDFLoader

# For backwards compatibility
TelegramChatLoader = TelegramChatFileLoader

__all__ = [
    "AcreomLoader",
    "AsyncHtmlLoader",
    "AsyncChromiumLoader",
    "AZLyricsLoader",
    "AcreomLoader",
    "AirbyteCDKLoader",
    "AirbyteGongLoader",
    "AirbyteJSONLoader",
    "AirbyteHubspotLoader",
    "AirbyteSalesforceLoader",
    "AirbyteShopifyLoader",
    "AirbyteStripeLoader",
    "AirbyteTypeformLoader",
    "AirbyteZendeskSupportLoader",
    "AirtableLoader",
    "AmazonTextractPDFLoader",
    "ApifyDatasetLoader",
    "ArcGISLoader",
    "ArxivLoader",
    "AssemblyAIAudioTranscriptLoader",
    "AsyncHtmlLoader",
    "AzureBlobStorageContainerLoader",
    "AzureBlobStorageFileLoader",
    "BSHTMLLoader",
    "BibtexLoader",
    "BigQueryLoader",
    "BiliBiliLoader",
    "BlackboardLoader",
    "Blob",
    "BlobLoader",
    "BlockchainDocumentLoader",
    "BraveSearchLoader",
    "BrowserlessLoader",
    "CSVLoader",
    "ChatGPTLoader",
    "CoNLLULoader",
    "CollegeConfidentialLoader",
    "ConcurrentLoader",
    "ConfluenceLoader",
    "CubeSemanticLoader",
    "DataFrameLoader",
    "DatadogLogsLoader",
    "DiffbotLoader",
    "DirectoryLoader",
    "DiscordChatLoader",
    "DocugamiLoader",
    "Docx2txtLoader",
    "DropboxLoader",
    "DuckDBLoader",
    "EmbaasBlobLoader",
    "EmbaasLoader",
    "EtherscanLoader",
    "EverNoteLoader",
    "FacebookChatLoader",
    "FaunaLoader",
    "FigmaFileLoader",
    "FileSystemBlobLoader",
    "GCSDirectoryLoader",
    "GCSFileLoader",
    "GeoDataFrameLoader",
    "GitHubIssuesLoader",
    "GitLoader",
    "GitbookLoader",
    "GoogleApiClient",
    "GoogleApiYoutubeLoader",
    "GoogleDriveLoader",
    "GutenbergLoader",
    "HNLoader",
    "HuggingFaceDatasetLoader",
    "IFixitLoader",
    "IMSDbLoader",
    "ImageCaptionLoader",
    "IuguLoader",
    "JSONLoader",
    "JoplinLoader",
    "LarkSuiteDocLoader",
    "MHTMLLoader",
    "MWDumpLoader",
    "MastodonTootsLoader",
    "MathpixPDFLoader",
    "MaxComputeLoader",
    "MergedDataLoader",
    "ModernTreasuryLoader",
    "MongodbLoader",
    "NewsURLLoader",
    "NotebookLoader",
    "NotionDBLoader",
    "NotionDirectoryLoader",
    "OBSDirectoryLoader",
    "OBSFileLoader",
    "ObsidianLoader",
    "OneDriveFileLoader",
    "OneDriveLoader",
    "OnlinePDFLoader",
    "OpenCityDataLoader",
    "OutlookMessageLoader",
    "PDFMinerLoader",
    "PDFMinerPDFasHTMLLoader",
    "PDFPlumberLoader",
    "PagedPDFSplitter",
    "PlaywrightURLLoader",
    "PolarsDataFrameLoader",
    "PsychicLoader",
    "PubMedLoader",
    "PyMuPDFLoader",
    "PyPDFDirectoryLoader",
    "PyPDFLoader",
    "PyPDFium2Loader",
    "PySparkDataFrameLoader",
    "PythonLoader",
    "RSSFeedLoader",
    "ReadTheDocsLoader",
    "RecursiveUrlLoader",
    "RedditPostsLoader",
    "RoamLoader",
    "RocksetLoader",
    "S3DirectoryLoader",
    "S3FileLoader",
    "SRTLoader",
    "SeleniumURLLoader",
    "SharePointLoader",
    "SitemapLoader",
    "SlackDirectoryLoader",
    "SnowflakeLoader",
    "SpreedlyLoader",
    "StripeLoader",
    "TelegramChatApiLoader",
    "TelegramChatFileLoader",
    "TelegramChatLoader",
    "TensorflowDatasetLoader",
    "TencentCOSDirectoryLoader",
    "TencentCOSFileLoader",
    "TextLoader",
    "ToMarkdownLoader",
    "TomlLoader",
    "TrelloLoader",
    "TwitterTweetLoader",
    "UnstructuredAPIFileIOLoader",
    "UnstructuredAPIFileLoader",
    "UnstructuredCSVLoader",
    "UnstructuredEPubLoader",
    "UnstructuredEmailLoader",
    "UnstructuredExcelLoader",
    "UnstructuredFileIOLoader",
    "UnstructuredFileLoader",
    "UnstructuredHTMLLoader",
    "UnstructuredImageLoader",
    "UnstructuredMarkdownLoader",
    "UnstructuredODTLoader",
    "UnstructuredOrgModeLoader",
    "UnstructuredPDFLoader",
    "UnstructuredPowerPointLoader",
    "UnstructuredRSTLoader",
    "UnstructuredRTFLoader",
    "UnstructuredTSVLoader",
    "UnstructuredURLLoader",
    "UnstructuredWordDocumentLoader",
    "UnstructuredXMLLoader",
    "WeatherDataLoader",
    "WebBaseLoader",
    "WhatsAppChatLoader",
    "WikipediaLoader",
    "XorbitsLoader",
    "YoutubeAudioLoader",
    "YoutubeLoader",
]
