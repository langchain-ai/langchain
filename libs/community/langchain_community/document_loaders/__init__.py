"""**Document Loaders**  are classes to load Documents.

**Document Loaders** are usually used to load a lot of Documents in a single run.

**Class hierarchy:**

.. code-block::

    BaseLoader --> <name>Loader  # Examples: TextLoader, UnstructuredFileLoader

**Main helpers:**

.. code-block::

    Document, <name>TextSplitter
"""
import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_community.document_loaders.acreom import (
        AcreomLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.airbyte import (
        AirbyteCDKLoader,  # noqa: F401
        AirbyteGongLoader,  # noqa: F401
        AirbyteHubspotLoader,  # noqa: F401
        AirbyteSalesforceLoader,  # noqa: F401
        AirbyteShopifyLoader,  # noqa: F401
        AirbyteStripeLoader,  # noqa: F401
        AirbyteTypeformLoader,  # noqa: F401
        AirbyteZendeskSupportLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.airbyte_json import (
        AirbyteJSONLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.airtable import (
        AirtableLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.apify_dataset import (
        ApifyDatasetLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.arcgis_loader import (
        ArcGISLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.arxiv import (
        ArxivLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.assemblyai import (
        AssemblyAIAudioLoaderById,  # noqa: F401
        AssemblyAIAudioTranscriptLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.astradb import (
        AstraDBLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.async_html import (
        AsyncHtmlLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.athena import (
        AthenaLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.azlyrics import (
        AZLyricsLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.azure_ai_data import (
        AzureAIDataLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.azure_blob_storage_container import (
        AzureBlobStorageContainerLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.azure_blob_storage_file import (
        AzureBlobStorageFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.bibtex import (
        BibtexLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.bigquery import (
        BigQueryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.bilibili import (
        BiliBiliLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.blackboard import (
        BlackboardLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.blob_loaders import (
        Blob,  # noqa: F401
        BlobLoader,  # noqa: F401
        FileSystemBlobLoader,  # noqa: F401
        YoutubeAudioLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.blockchain import (
        BlockchainDocumentLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.brave_search import (
        BraveSearchLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.browserless import (
        BrowserlessLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.cassandra import (
        CassandraLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.chatgpt import (
        ChatGPTLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.chm import (
        UnstructuredCHMLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.chromium import (
        AsyncChromiumLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.college_confidential import (
        CollegeConfidentialLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.concurrent import (
        ConcurrentLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.confluence import (
        ConfluenceLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.conllu import (
        CoNLLULoader,  # noqa: F401
    )
    from langchain_community.document_loaders.couchbase import (
        CouchbaseLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.csv_loader import (
        CSVLoader,  # noqa: F401
        UnstructuredCSVLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.cube_semantic import (
        CubeSemanticLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.datadog_logs import (
        DatadogLogsLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.dataframe import (
        DataFrameLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.diffbot import (
        DiffbotLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.directory import (
        DirectoryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.discord import (
        DiscordChatLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.doc_intelligence import (
        AzureAIDocumentIntelligenceLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.docugami import (
        DocugamiLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.docusaurus import (
        DocusaurusLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.dropbox import (
        DropboxLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.duckdb_loader import (
        DuckDBLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.email import (
        OutlookMessageLoader,  # noqa: F401
        UnstructuredEmailLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.epub import (
        UnstructuredEPubLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.etherscan import (
        EtherscanLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.evernote import (
        EverNoteLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.excel import (
        UnstructuredExcelLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.facebook_chat import (
        FacebookChatLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.fauna import (
        FaunaLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.figma import (
        FigmaFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.firecrawl import (
        FireCrawlLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.gcs_directory import (
        GCSDirectoryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.gcs_file import (
        GCSFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.geodataframe import (
        GeoDataFrameLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.git import (
        GitLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.gitbook import (
        GitbookLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.github import (
        GithubFileLoader,  # noqa: F401
        GitHubIssuesLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.glue_catalog import (
        GlueCatalogLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.google_speech_to_text import (
        GoogleSpeechToTextLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.googledrive import (
        GoogleDriveLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.gutenberg import (
        GutenbergLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.hn import (
        HNLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.html import (
        UnstructuredHTMLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.html_bs import (
        BSHTMLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.hugging_face_dataset import (
        HuggingFaceDatasetLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.hugging_face_model import (
        HuggingFaceModelLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.ifixit import (
        IFixitLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.image import (
        UnstructuredImageLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.image_captions import (
        ImageCaptionLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.imsdb import (
        IMSDbLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.iugu import (
        IuguLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.joplin import (
        JoplinLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.json_loader import (
        JSONLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.lakefs import (
        LakeFSLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.larksuite import (
        LarkSuiteDocLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.llmsherpa import (
        LLMSherpaFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.markdown import (
        UnstructuredMarkdownLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.mastodon import (
        MastodonTootsLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.max_compute import (
        MaxComputeLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.mediawikidump import (
        MWDumpLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.merge import (
        MergedDataLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.mhtml import (
        MHTMLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.modern_treasury import (
        ModernTreasuryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.mongodb import (
        MongodbLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.news import (
        NewsURLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.notebook import (
        NotebookLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.notion import (
        NotionDirectoryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.notiondb import (
        NotionDBLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.obs_directory import (
        OBSDirectoryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.obs_file import (
        OBSFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.obsidian import (
        ObsidianLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.odt import (
        UnstructuredODTLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.onedrive import (
        OneDriveLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.onedrive_file import (
        OneDriveFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.open_city_data import (
        OpenCityDataLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.oracleadb_loader import (
        OracleAutonomousDatabaseLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.org_mode import (
        UnstructuredOrgModeLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.pdf import (
        AmazonTextractPDFLoader,  # noqa: F401
        MathpixPDFLoader,  # noqa: F401
        OnlinePDFLoader,  # noqa: F401
        PagedPDFSplitter,  # noqa: F401
        PDFMinerLoader,  # noqa: F401
        PDFMinerPDFasHTMLLoader,  # noqa: F401
        PDFPlumberLoader,  # noqa: F401
        PyMuPDFLoader,  # noqa: F401
        PyPDFDirectoryLoader,  # noqa: F401
        PyPDFium2Loader,  # noqa: F401
        PyPDFLoader,  # noqa: F401
        UnstructuredPDFLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.pebblo import (
        PebbloSafeLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.polars_dataframe import (
        PolarsDataFrameLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.powerpoint import (
        UnstructuredPowerPointLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.psychic import (
        PsychicLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.pubmed import (
        PubMedLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.pyspark_dataframe import (
        PySparkDataFrameLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.python import (
        PythonLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.readthedocs import (
        ReadTheDocsLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.recursive_url_loader import (
        RecursiveUrlLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.reddit import (
        RedditPostsLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.roam import (
        RoamLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.rocksetdb import (
        RocksetLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.rss import (
        RSSFeedLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.rst import (
        UnstructuredRSTLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.rtf import (
        UnstructuredRTFLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.s3_directory import (
        S3DirectoryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.s3_file import (
        S3FileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.sharepoint import (
        SharePointLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.sitemap import (
        SitemapLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.slack_directory import (
        SlackDirectoryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.snowflake_loader import (
        SnowflakeLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.spreedly import (
        SpreedlyLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.sql_database import (
        SQLDatabaseLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.srt import (
        SRTLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.stripe import (
        StripeLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.surrealdb import (
        SurrealDBLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.telegram import (
        TelegramChatApiLoader,  # noqa: F401
        TelegramChatFileLoader,  # noqa: F401
        TelegramChatLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.tencent_cos_directory import (
        TencentCOSDirectoryLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.tencent_cos_file import (
        TencentCOSFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.tensorflow_datasets import (
        TensorflowDatasetLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.text import (
        TextLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.tidb import (
        TiDBLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.tomarkdown import (
        ToMarkdownLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.toml import (
        TomlLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.trello import (
        TrelloLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.tsv import (
        UnstructuredTSVLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.twitter import (
        TwitterTweetLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.unstructured import (
        UnstructuredAPIFileIOLoader,  # noqa: F401
        UnstructuredAPIFileLoader,  # noqa: F401
        UnstructuredFileIOLoader,  # noqa: F401
        UnstructuredFileLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.url import (
        UnstructuredURLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.url_playwright import (
        PlaywrightURLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.url_selenium import (
        SeleniumURLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.vsdx import (
        VsdxLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.weather import (
        WeatherDataLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.web_base import (
        WebBaseLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.whatsapp_chat import (
        WhatsAppChatLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.wikipedia import (
        WikipediaLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.word_document import (
        Docx2txtLoader,  # noqa: F401
        UnstructuredWordDocumentLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.xml import (
        UnstructuredXMLLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.xorbits import (
        XorbitsLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.youtube import (
        GoogleApiClient,  # noqa: F401
        GoogleApiYoutubeLoader,  # noqa: F401
        YoutubeLoader,  # noqa: F401
    )
    from langchain_community.document_loaders.yuque import (
        YuqueLoader,  # noqa: F401
    )

__all__ = [
    "AZLyricsLoader",
    "AcreomLoader",
    "AirbyteCDKLoader",
    "AirbyteGongLoader",
    "AirbyteHubspotLoader",
    "AirbyteJSONLoader",
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
    "AssemblyAIAudioLoaderById",
    "AssemblyAIAudioTranscriptLoader",
    "AstraDBLoader",
    "AsyncChromiumLoader",
    "AsyncHtmlLoader",
    "AthenaLoader",
    "AzureAIDataLoader",
    "AzureAIDocumentIntelligenceLoader",
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
    "CassandraLoader",
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
    "FireCrawlLoader",
    "FileSystemBlobLoader",
    "GCSDirectoryLoader",
    "GCSFileLoader",
    "GeoDataFrameLoader",
    "GitHubIssuesLoader",
    "GitLoader",
    "GitbookLoader",
    "GithubFileLoader",
    "GoogleApiClient",
    "GoogleApiYoutubeLoader",
    "GoogleDriveLoader",
    "GoogleSpeechToTextLoader",
    "GutenbergLoader",
    "HNLoader",
    "HuggingFaceDatasetLoader",
    "HuggingFaceModelLoader",
    "IFixitLoader",
    "IMSDbLoader",
    "ImageCaptionLoader",
    "IuguLoader",
    "JSONLoader",
    "JoplinLoader",
    "LLMSherpaFileLoader",
    "LakeFSLoader",
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
    "OracleAutonomousDatabaseLoader",
    "OutlookMessageLoader",
    "PDFMinerLoader",
    "PDFMinerPDFasHTMLLoader",
    "PDFPlumberLoader",
    "PagedPDFSplitter",
    "PebbloSafeLoader",
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
    "SQLDatabaseLoader",
    "SRTLoader",
    "SeleniumURLLoader",
    "SharePointLoader",
    "SitemapLoader",
    "SlackDirectoryLoader",
    "SnowflakeLoader",
    "SpreedlyLoader",
    "StripeLoader",
    "SurrealDBLoader",
    "TelegramChatApiLoader",
    "TelegramChatFileLoader",
    "TelegramChatLoader",
    "TencentCOSDirectoryLoader",
    "TencentCOSFileLoader",
    "TensorflowDatasetLoader",
    "TextLoader",
    "TiDBLoader",
    "ToMarkdownLoader",
    "TomlLoader",
    "TrelloLoader",
    "TwitterTweetLoader",
    "UnstructuredAPIFileIOLoader",
    "UnstructuredAPIFileLoader",
    "UnstructuredCHMLoader",
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
    "VsdxLoader",
    "WeatherDataLoader",
    "WebBaseLoader",
    "WhatsAppChatLoader",
    "WikipediaLoader",
    "XorbitsLoader",
    "YoutubeAudioLoader",
    "YoutubeLoader",
    "YuqueLoader",
]

_module_lookup = {
    "AZLyricsLoader": "langchain_community.document_loaders.azlyrics",
    "AcreomLoader": "langchain_community.document_loaders.acreom",
    "AirbyteCDKLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteGongLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteHubspotLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteJSONLoader": "langchain_community.document_loaders.airbyte_json",
    "AirbyteSalesforceLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteShopifyLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteStripeLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteTypeformLoader": "langchain_community.document_loaders.airbyte",
    "AirbyteZendeskSupportLoader": "langchain_community.document_loaders.airbyte",
    "AirtableLoader": "langchain_community.document_loaders.airtable",
    "AmazonTextractPDFLoader": "langchain_community.document_loaders.pdf",
    "ApifyDatasetLoader": "langchain_community.document_loaders.apify_dataset",
    "ArcGISLoader": "langchain_community.document_loaders.arcgis_loader",
    "ArxivLoader": "langchain_community.document_loaders.arxiv",
    "AssemblyAIAudioLoaderById": "langchain_community.document_loaders.assemblyai",
    "AssemblyAIAudioTranscriptLoader": "langchain_community.document_loaders.assemblyai",  # noqa: E501
    "AstraDBLoader": "langchain_community.document_loaders.astradb",
    "AsyncChromiumLoader": "langchain_community.document_loaders.chromium",
    "AsyncHtmlLoader": "langchain_community.document_loaders.async_html",
    "AthenaLoader": "langchain_community.document_loaders.athena",
    "AzureAIDataLoader": "langchain_community.document_loaders.azure_ai_data",
    "AzureAIDocumentIntelligenceLoader": "langchain_community.document_loaders.doc_intelligence",  # noqa: E501
    "AzureBlobStorageContainerLoader": "langchain_community.document_loaders.azure_blob_storage_container",  # noqa: E501
    "AzureBlobStorageFileLoader": "langchain_community.document_loaders.azure_blob_storage_file",  # noqa: E501
    "BSHTMLLoader": "langchain_community.document_loaders.html_bs",
    "BibtexLoader": "langchain_community.document_loaders.bibtex",
    "BigQueryLoader": "langchain_community.document_loaders.bigquery",
    "BiliBiliLoader": "langchain_community.document_loaders.bilibili",
    "BlackboardLoader": "langchain_community.document_loaders.blackboard",
    "Blob": "langchain_community.document_loaders.blob_loaders",
    "BlobLoader": "langchain_community.document_loaders.blob_loaders",
    "BlockchainDocumentLoader": "langchain_community.document_loaders.blockchain",
    "BraveSearchLoader": "langchain_community.document_loaders.brave_search",
    "BrowserlessLoader": "langchain_community.document_loaders.browserless",
    "CSVLoader": "langchain_community.document_loaders.csv_loader",
    "CassandraLoader": "langchain_community.document_loaders.cassandra",
    "ChatGPTLoader": "langchain_community.document_loaders.chatgpt",
    "CoNLLULoader": "langchain_community.document_loaders.conllu",
    "CollegeConfidentialLoader": "langchain_community.document_loaders.college_confidential",  # noqa: E501
    "ConcurrentLoader": "langchain_community.document_loaders.concurrent",
    "ConfluenceLoader": "langchain_community.document_loaders.confluence",
    "CouchbaseLoader": "langchain_community.document_loaders.couchbase",
    "CubeSemanticLoader": "langchain_community.document_loaders.cube_semantic",
    "DataFrameLoader": "langchain_community.document_loaders.dataframe",
    "DatadogLogsLoader": "langchain_community.document_loaders.datadog_logs",
    "DiffbotLoader": "langchain_community.document_loaders.diffbot",
    "DirectoryLoader": "langchain_community.document_loaders.directory",
    "DiscordChatLoader": "langchain_community.document_loaders.discord",
    "DocugamiLoader": "langchain_community.document_loaders.docugami",
    "DocusaurusLoader": "langchain_community.document_loaders.docusaurus",
    "Docx2txtLoader": "langchain_community.document_loaders.word_document",
    "DropboxLoader": "langchain_community.document_loaders.dropbox",
    "DuckDBLoader": "langchain_community.document_loaders.duckdb_loader",
    "EtherscanLoader": "langchain_community.document_loaders.etherscan",
    "EverNoteLoader": "langchain_community.document_loaders.evernote",
    "FacebookChatLoader": "langchain_community.document_loaders.facebook_chat",
    "FaunaLoader": "langchain_community.document_loaders.fauna",
    "FigmaFileLoader": "langchain_community.document_loaders.figma",
    "FireCrawlLoader": "langchain_community.document_loaders.firecrawl",
    "FileSystemBlobLoader": "langchain_community.document_loaders.blob_loaders",
    "GCSDirectoryLoader": "langchain_community.document_loaders.gcs_directory",
    "GCSFileLoader": "langchain_community.document_loaders.gcs_file",
    "GeoDataFrameLoader": "langchain_community.document_loaders.geodataframe",
    "GitHubIssuesLoader": "langchain_community.document_loaders.github",
    "GitLoader": "langchain_community.document_loaders.git",
    "GitbookLoader": "langchain_community.document_loaders.gitbook",
    "GithubFileLoader": "langchain_community.document_loaders.github",
    "GlueCatalogLoader": "langchain_community.document_loaders.glue_catalog",
    "GoogleApiClient": "langchain_community.document_loaders.youtube",
    "GoogleApiYoutubeLoader": "langchain_community.document_loaders.youtube",
    "GoogleDriveLoader": "langchain_community.document_loaders.googledrive",
    "GoogleSpeechToTextLoader": "langchain_community.document_loaders.google_speech_to_text",  # noqa: E501
    "GutenbergLoader": "langchain_community.document_loaders.gutenberg",
    "HNLoader": "langchain_community.document_loaders.hn",
    "HuggingFaceDatasetLoader": "langchain_community.document_loaders.hugging_face_dataset",  # noqa: E501
    "HuggingFaceModelLoader": "langchain_community.document_loaders.hugging_face_model",
    "IFixitLoader": "langchain_community.document_loaders.ifixit",
    "IMSDbLoader": "langchain_community.document_loaders.imsdb",
    "ImageCaptionLoader": "langchain_community.document_loaders.image_captions",
    "IuguLoader": "langchain_community.document_loaders.iugu",
    "JSONLoader": "langchain_community.document_loaders.json_loader",
    "JoplinLoader": "langchain_community.document_loaders.joplin",
    "LakeFSLoader": "langchain_community.document_loaders.lakefs",
    "LarkSuiteDocLoader": "langchain_community.document_loaders.larksuite",
    "LLMSherpaFileLoader": "langchain_community.document_loaders.llmsherpa",
    "MHTMLLoader": "langchain_community.document_loaders.mhtml",
    "MWDumpLoader": "langchain_community.document_loaders.mediawikidump",
    "MastodonTootsLoader": "langchain_community.document_loaders.mastodon",
    "MathpixPDFLoader": "langchain_community.document_loaders.pdf",
    "MaxComputeLoader": "langchain_community.document_loaders.max_compute",
    "MergedDataLoader": "langchain_community.document_loaders.merge",
    "ModernTreasuryLoader": "langchain_community.document_loaders.modern_treasury",
    "MongodbLoader": "langchain_community.document_loaders.mongodb",
    "NewsURLLoader": "langchain_community.document_loaders.news",
    "NotebookLoader": "langchain_community.document_loaders.notebook",
    "NotionDBLoader": "langchain_community.document_loaders.notiondb",
    "NotionDirectoryLoader": "langchain_community.document_loaders.notion",
    "OBSDirectoryLoader": "langchain_community.document_loaders.obs_directory",
    "OBSFileLoader": "langchain_community.document_loaders.obs_file",
    "ObsidianLoader": "langchain_community.document_loaders.obsidian",
    "OneDriveFileLoader": "langchain_community.document_loaders.onedrive_file",
    "OneDriveLoader": "langchain_community.document_loaders.onedrive",
    "OnlinePDFLoader": "langchain_community.document_loaders.pdf",
    "OpenCityDataLoader": "langchain_community.document_loaders.open_city_data",
    "OracleAutonomousDatabaseLoader": "langchain_community.document_loaders.oracleadb_loader",  # noqa: E501
    "OutlookMessageLoader": "langchain_community.document_loaders.email",
    "PDFMinerLoader": "langchain_community.document_loaders.pdf",
    "PDFMinerPDFasHTMLLoader": "langchain_community.document_loaders.pdf",
    "PDFPlumberLoader": "langchain_community.document_loaders.pdf",
    "PagedPDFSplitter": "langchain_community.document_loaders.pdf",
    "PebbloSafeLoader": "langchain_community.document_loaders.pebblo",
    "PlaywrightURLLoader": "langchain_community.document_loaders.url_playwright",
    "PolarsDataFrameLoader": "langchain_community.document_loaders.polars_dataframe",
    "PsychicLoader": "langchain_community.document_loaders.psychic",
    "PubMedLoader": "langchain_community.document_loaders.pubmed",
    "PyMuPDFLoader": "langchain_community.document_loaders.pdf",
    "PyPDFDirectoryLoader": "langchain_community.document_loaders.pdf",
    "PyPDFLoader": "langchain_community.document_loaders.pdf",
    "PyPDFium2Loader": "langchain_community.document_loaders.pdf",
    "PySparkDataFrameLoader": "langchain_community.document_loaders.pyspark_dataframe",
    "PythonLoader": "langchain_community.document_loaders.python",
    "RSSFeedLoader": "langchain_community.document_loaders.rss",
    "ReadTheDocsLoader": "langchain_community.document_loaders.readthedocs",
    "RecursiveUrlLoader": "langchain_community.document_loaders.recursive_url_loader",
    "RedditPostsLoader": "langchain_community.document_loaders.reddit",
    "RoamLoader": "langchain_community.document_loaders.roam",
    "RocksetLoader": "langchain_community.document_loaders.rocksetdb",
    "S3DirectoryLoader": "langchain_community.document_loaders.s3_directory",
    "S3FileLoader": "langchain_community.document_loaders.s3_file",
    "SQLDatabaseLoader": "langchain_community.document_loaders.sql_database",
    "SRTLoader": "langchain_community.document_loaders.srt",
    "SeleniumURLLoader": "langchain_community.document_loaders.url_selenium",
    "SharePointLoader": "langchain_community.document_loaders.sharepoint",
    "SitemapLoader": "langchain_community.document_loaders.sitemap",
    "SlackDirectoryLoader": "langchain_community.document_loaders.slack_directory",
    "SnowflakeLoader": "langchain_community.document_loaders.snowflake_loader",
    "SpreedlyLoader": "langchain_community.document_loaders.spreedly",
    "StripeLoader": "langchain_community.document_loaders.stripe",
    "SurrealDBLoader": "langchain_community.document_loaders.surrealdb",
    "TelegramChatApiLoader": "langchain_community.document_loaders.telegram",
    "TelegramChatFileLoader": "langchain_community.document_loaders.telegram",
    "TelegramChatLoader": "langchain_community.document_loaders.telegram",
    "TencentCOSDirectoryLoader": "langchain_community.document_loaders.tencent_cos_directory",  # noqa: E501
    "TencentCOSFileLoader": "langchain_community.document_loaders.tencent_cos_file",
    "TensorflowDatasetLoader": "langchain_community.document_loaders.tensorflow_datasets",  # noqa: E501
    "TextLoader": "langchain_community.document_loaders.text",
    "TiDBLoader": "langchain_community.document_loaders.tidb",
    "ToMarkdownLoader": "langchain_community.document_loaders.tomarkdown",
    "TomlLoader": "langchain_community.document_loaders.toml",
    "TrelloLoader": "langchain_community.document_loaders.trello",
    "TwitterTweetLoader": "langchain_community.document_loaders.twitter",
    "UnstructuredAPIFileIOLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredAPIFileLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredCHMLoader": "langchain_community.document_loaders.chm",
    "UnstructuredCSVLoader": "langchain_community.document_loaders.csv_loader",
    "UnstructuredEPubLoader": "langchain_community.document_loaders.epub",
    "UnstructuredEmailLoader": "langchain_community.document_loaders.email",
    "UnstructuredExcelLoader": "langchain_community.document_loaders.excel",
    "UnstructuredFileIOLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredFileLoader": "langchain_community.document_loaders.unstructured",
    "UnstructuredHTMLLoader": "langchain_community.document_loaders.html",
    "UnstructuredImageLoader": "langchain_community.document_loaders.image",
    "UnstructuredMarkdownLoader": "langchain_community.document_loaders.markdown",
    "UnstructuredODTLoader": "langchain_community.document_loaders.odt",
    "UnstructuredOrgModeLoader": "langchain_community.document_loaders.org_mode",
    "UnstructuredPDFLoader": "langchain_community.document_loaders.pdf",
    "UnstructuredPowerPointLoader": "langchain_community.document_loaders.powerpoint",
    "UnstructuredRSTLoader": "langchain_community.document_loaders.rst",
    "UnstructuredRTFLoader": "langchain_community.document_loaders.rtf",
    "UnstructuredTSVLoader": "langchain_community.document_loaders.tsv",
    "UnstructuredURLLoader": "langchain_community.document_loaders.url",
    "UnstructuredWordDocumentLoader": "langchain_community.document_loaders.word_document",  # noqa: E501
    "UnstructuredXMLLoader": "langchain_community.document_loaders.xml",
    "VsdxLoader": "langchain_community.document_loaders.vsdx",
    "WeatherDataLoader": "langchain_community.document_loaders.weather",
    "WebBaseLoader": "langchain_community.document_loaders.web_base",
    "WhatsAppChatLoader": "langchain_community.document_loaders.whatsapp_chat",
    "WikipediaLoader": "langchain_community.document_loaders.wikipedia",
    "XorbitsLoader": "langchain_community.document_loaders.xorbits",
    "YoutubeAudioLoader": "langchain_community.document_loaders.blob_loaders",
    "YoutubeLoader": "langchain_community.document_loaders.youtube",
    "YuqueLoader": "langchain_community.document_loaders.yuque",
}


def __getattr__(name: str) -> Any:
    if name in _module_lookup:
        module = importlib.import_module(_module_lookup[name])
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")


__all__ = list(_module_lookup.keys())
