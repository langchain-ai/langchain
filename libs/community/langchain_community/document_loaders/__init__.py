"""**Document Loaders**  are classes to load Documents.

**Document Loaders** are usually used to load a lot of Documents in a single run.

**Class hierarchy:**

.. code-block::

    BaseLoader --> <name>Loader  # Examples: TextLoader, UnstructuredFileLoader

**Main helpers:**

.. code-block::

    Document, <name>TextSplitter
"""

from langchain_community.document_loaders.acreom import AcreomLoader
from langchain_community.document_loaders.airbyte import (
    AirbyteCDKLoader,
    AirbyteGongLoader,
    AirbyteHubspotLoader,
    AirbyteSalesforceLoader,
    AirbyteShopifyLoader,
    AirbyteStripeLoader,
    AirbyteTypeformLoader,
    AirbyteZendeskSupportLoader,
)
from langchain_community.document_loaders.airbyte_json import AirbyteJSONLoader
from langchain_community.document_loaders.airtable import AirtableLoader
from langchain_community.document_loaders.apify_dataset import ApifyDatasetLoader
from langchain_community.document_loaders.arcgis_loader import ArcGISLoader
from langchain_community.document_loaders.arxiv import ArxivLoader
from langchain_community.document_loaders.assemblyai import (
    AssemblyAIAudioTranscriptLoader,
)
from langchain_community.document_loaders.async_html import AsyncHtmlLoader
from langchain_community.document_loaders.azlyrics import AZLyricsLoader
from langchain_community.document_loaders.azure_ai_data import (
    AzureAIDataLoader,
)
from langchain_community.document_loaders.azure_blob_storage_container import (
    AzureBlobStorageContainerLoader,
)
from langchain_community.document_loaders.azure_blob_storage_file import (
    AzureBlobStorageFileLoader,
)
from langchain_community.document_loaders.bibtex import BibtexLoader
from langchain_community.document_loaders.bigquery import BigQueryLoader
from langchain_community.document_loaders.bilibili import BiliBiliLoader
from langchain_community.document_loaders.blackboard import BlackboardLoader
from langchain_community.document_loaders.blob_loaders import (
    Blob,
    BlobLoader,
    FileSystemBlobLoader,
    YoutubeAudioLoader,
)
from langchain_community.document_loaders.blockchain import BlockchainDocumentLoader
from langchain_community.document_loaders.brave_search import BraveSearchLoader
from langchain_community.document_loaders.browserless import BrowserlessLoader
from langchain_community.document_loaders.chatgpt import ChatGPTLoader
from langchain_community.document_loaders.chromium import AsyncChromiumLoader
from langchain_community.document_loaders.college_confidential import (
    CollegeConfidentialLoader,
)
from langchain_community.document_loaders.concurrent import ConcurrentLoader
from langchain_community.document_loaders.confluence import ConfluenceLoader
from langchain_community.document_loaders.conllu import CoNLLULoader
from langchain_community.document_loaders.couchbase import CouchbaseLoader
from langchain_community.document_loaders.csv_loader import (
    CSVLoader,
    UnstructuredCSVLoader,
)
from langchain_community.document_loaders.cube_semantic import CubeSemanticLoader
from langchain_community.document_loaders.datadog_logs import DatadogLogsLoader
from langchain_community.document_loaders.dataframe import DataFrameLoader
from langchain_community.document_loaders.diffbot import DiffbotLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.document_loaders.discord import DiscordChatLoader
from langchain_community.document_loaders.docugami import DocugamiLoader
from langchain_community.document_loaders.docusaurus import DocusaurusLoader
from langchain_community.document_loaders.dropbox import DropboxLoader
from langchain_community.document_loaders.duckdb_loader import DuckDBLoader
from langchain_community.document_loaders.email import (
    OutlookMessageLoader,
    UnstructuredEmailLoader,
)
from langchain_community.document_loaders.epub import UnstructuredEPubLoader
from langchain_community.document_loaders.etherscan import EtherscanLoader
from langchain_community.document_loaders.evernote import EverNoteLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders.facebook_chat import FacebookChatLoader
from langchain_community.document_loaders.fauna import FaunaLoader
from langchain_community.document_loaders.figma import FigmaFileLoader
from langchain_community.document_loaders.gcs_directory import GCSDirectoryLoader
from langchain_community.document_loaders.gcs_file import GCSFileLoader
from langchain_community.document_loaders.geodataframe import GeoDataFrameLoader
from langchain_community.document_loaders.git import GitLoader
from langchain_community.document_loaders.gitbook import GitbookLoader
from langchain_community.document_loaders.github import GitHubIssuesLoader
from langchain_community.document_loaders.google_speech_to_text import (
    GoogleSpeechToTextLoader,
)
from langchain_community.document_loaders.googledrive import GoogleDriveLoader
from langchain_community.document_loaders.gutenberg import GutenbergLoader
from langchain_community.document_loaders.hn import HNLoader
from langchain_community.document_loaders.html import UnstructuredHTMLLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.hugging_face_dataset import (
    HuggingFaceDatasetLoader,
)
from langchain_community.document_loaders.ifixit import IFixitLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain_community.document_loaders.image_captions import ImageCaptionLoader
from langchain_community.document_loaders.imsdb import IMSDbLoader
from langchain_community.document_loaders.iugu import IuguLoader
from langchain_community.document_loaders.joplin import JoplinLoader
from langchain_community.document_loaders.json_loader import JSONLoader
from langchain_community.document_loaders.lakefs import LakeFSLoader
from langchain_community.document_loaders.larksuite import LarkSuiteDocLoader
from langchain_community.document_loaders.markdown import UnstructuredMarkdownLoader
from langchain_community.document_loaders.mastodon import MastodonTootsLoader
from langchain_community.document_loaders.max_compute import MaxComputeLoader
from langchain_community.document_loaders.mediawikidump import MWDumpLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain_community.document_loaders.mhtml import MHTMLLoader
from langchain_community.document_loaders.modern_treasury import ModernTreasuryLoader
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain_community.document_loaders.news import NewsURLLoader
from langchain_community.document_loaders.notebook import NotebookLoader
from langchain_community.document_loaders.notion import NotionDirectoryLoader
from langchain_community.document_loaders.notiondb import NotionDBLoader
from langchain_community.document_loaders.obs_directory import OBSDirectoryLoader
from langchain_community.document_loaders.obs_file import OBSFileLoader
from langchain_community.document_loaders.obsidian import ObsidianLoader
from langchain_community.document_loaders.odt import UnstructuredODTLoader
from langchain_community.document_loaders.onedrive import OneDriveLoader
from langchain_community.document_loaders.onedrive_file import OneDriveFileLoader
from langchain_community.document_loaders.open_city_data import OpenCityDataLoader
from langchain_community.document_loaders.org_mode import UnstructuredOrgModeLoader
from langchain_community.document_loaders.pdf import (
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
from langchain_community.document_loaders.polars_dataframe import PolarsDataFrameLoader
from langchain_community.document_loaders.powerpoint import UnstructuredPowerPointLoader
from langchain_community.document_loaders.psychic import PsychicLoader
from langchain_community.document_loaders.pubmed import PubMedLoader
from langchain_community.document_loaders.pyspark_dataframe import (
    PySparkDataFrameLoader,
)
from langchain_community.document_loaders.python import PythonLoader
from langchain_community.document_loaders.readthedocs import ReadTheDocsLoader
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain_community.document_loaders.reddit import RedditPostsLoader
from langchain_community.document_loaders.roam import RoamLoader
from langchain_community.document_loaders.rocksetdb import RocksetLoader
from langchain_community.document_loaders.rss import RSSFeedLoader
from langchain_community.document_loaders.rst import UnstructuredRSTLoader
from langchain_community.document_loaders.rtf import UnstructuredRTFLoader
from langchain_community.document_loaders.s3_directory import S3DirectoryLoader
from langchain_community.document_loaders.s3_file import S3FileLoader
from langchain_community.document_loaders.sharepoint import SharePointLoader
from langchain_community.document_loaders.sitemap import SitemapLoader
from langchain_community.document_loaders.slack_directory import SlackDirectoryLoader
from langchain_community.document_loaders.snowflake_loader import SnowflakeLoader
from langchain_community.document_loaders.spreedly import SpreedlyLoader
from langchain_community.document_loaders.srt import SRTLoader
from langchain_community.document_loaders.stripe import StripeLoader
from langchain_community.document_loaders.telegram import (
    TelegramChatApiLoader,
    TelegramChatFileLoader,
)
from langchain_community.document_loaders.tencent_cos_directory import (
    TencentCOSDirectoryLoader,
)
from langchain_community.document_loaders.tencent_cos_file import TencentCOSFileLoader
from langchain_community.document_loaders.tensorflow_datasets import (
    TensorflowDatasetLoader,
)
from langchain_community.document_loaders.text import TextLoader
from langchain_community.document_loaders.tomarkdown import ToMarkdownLoader
from langchain_community.document_loaders.toml import TomlLoader
from langchain_community.document_loaders.trello import TrelloLoader
from langchain_community.document_loaders.tsv import UnstructuredTSVLoader
from langchain_community.document_loaders.twitter import TwitterTweetLoader
from langchain_community.document_loaders.unstructured import (
    UnstructuredAPIFileIOLoader,
    UnstructuredAPIFileLoader,
    UnstructuredFileIOLoader,
    UnstructuredFileLoader,
)
from langchain_community.document_loaders.url import UnstructuredURLLoader
from langchain_community.document_loaders.url_playwright import PlaywrightURLLoader
from langchain_community.document_loaders.url_selenium import SeleniumURLLoader
from langchain_community.document_loaders.weather import WeatherDataLoader
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.whatsapp_chat import WhatsAppChatLoader
from langchain_community.document_loaders.wikipedia import WikipediaLoader
from langchain_community.document_loaders.word_document import (
    Docx2txtLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_community.document_loaders.xml import UnstructuredXMLLoader
from langchain_community.document_loaders.xorbits import XorbitsLoader
from langchain_community.document_loaders.youtube import (
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
    "AzureAIDataLoader",
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
    "CouchbaseLoader",
    "CubeSemanticLoader",
    "DataFrameLoader",
    "DatadogLogsLoader",
    "DiffbotLoader",
    "DirectoryLoader",
    "DiscordChatLoader",
    "DocugamiLoader",
    "DocusaurusLoader",
    "Docx2txtLoader",
    "DropboxLoader",
    "DuckDBLoader",
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
    "GoogleSpeechToTextLoader",
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
    "LakeFSLoader",
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
