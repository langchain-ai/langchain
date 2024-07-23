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
        AcreomLoader,
    )
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
    from langchain_community.document_loaders.airbyte_json import (
        AirbyteJSONLoader,
    )
    from langchain_community.document_loaders.airtable import (
        AirtableLoader,
    )
    from langchain_community.document_loaders.apify_dataset import (
        ApifyDatasetLoader,
    )
    from langchain_community.document_loaders.arcgis_loader import (
        ArcGISLoader,
    )
    from langchain_community.document_loaders.arxiv import (
        ArxivLoader,
    )
    from langchain_community.document_loaders.assemblyai import (
        AssemblyAIAudioLoaderById,
        AssemblyAIAudioTranscriptLoader,
    )
    from langchain_community.document_loaders.astradb import (
        AstraDBLoader,
    )
    from langchain_community.document_loaders.async_html import (
        AsyncHtmlLoader,
    )
    from langchain_community.document_loaders.athena import (
        AthenaLoader,
    )
    from langchain_community.document_loaders.azlyrics import (
        AZLyricsLoader,
    )
    from langchain_community.document_loaders.azure_ai_data import (
        AzureAIDataLoader,
    )
    from langchain_community.document_loaders.azure_blob_storage_container import (
        AzureBlobStorageContainerLoader,
    )
    from langchain_community.document_loaders.azure_blob_storage_file import (
        AzureBlobStorageFileLoader,
    )
    from langchain_community.document_loaders.bibtex import (
        BibtexLoader,
    )
    from langchain_community.document_loaders.bigquery import (
        BigQueryLoader,
    )
    from langchain_community.document_loaders.bilibili import (
        BiliBiliLoader,
    )
    from langchain_community.document_loaders.blackboard import (
        BlackboardLoader,
    )
    from langchain_community.document_loaders.blob_loaders import (
        Blob,
        BlobLoader,
        FileSystemBlobLoader,
        YoutubeAudioLoader,
    )
    from langchain_community.document_loaders.blockchain import (
        BlockchainDocumentLoader,
    )
    from langchain_community.document_loaders.brave_search import (
        BraveSearchLoader,
    )
    from langchain_community.document_loaders.browserbase import (
        BrowserbaseLoader,
    )
    from langchain_community.document_loaders.browserless import (
        BrowserlessLoader,
    )
    from langchain_community.document_loaders.cassandra import (
        CassandraLoader,
    )
    from langchain_community.document_loaders.chatgpt import (
        ChatGPTLoader,
    )
    from langchain_community.document_loaders.chm import (
        UnstructuredCHMLoader,
    )
    from langchain_community.document_loaders.chromium import (
        AsyncChromiumLoader,
    )
    from langchain_community.document_loaders.college_confidential import (
        CollegeConfidentialLoader,
    )
    from langchain_community.document_loaders.concurrent import (
        ConcurrentLoader,
    )
    from langchain_community.document_loaders.confluence import (
        ConfluenceLoader,
    )
    from langchain_community.document_loaders.conllu import (
        CoNLLULoader,
    )
    from langchain_community.document_loaders.couchbase import (
        CouchbaseLoader,
    )
    from langchain_community.document_loaders.csv_loader import (
        CSVLoader,
        UnstructuredCSVLoader,
    )
    from langchain_community.document_loaders.cube_semantic import (
        CubeSemanticLoader,
    )
    from langchain_community.document_loaders.datadog_logs import (
        DatadogLogsLoader,
    )
    from langchain_community.document_loaders.dataframe import (
        DataFrameLoader,
    )
    from langchain_community.document_loaders.dedoc import (
        DedocAPIFileLoader,
        DedocFileLoader,
    )
    from langchain_community.document_loaders.diffbot import (
        DiffbotLoader,
    )
    from langchain_community.document_loaders.directory import (
        DirectoryLoader,
    )
    from langchain_community.document_loaders.discord import (
        DiscordChatLoader,
    )
    from langchain_community.document_loaders.doc_intelligence import (
        AzureAIDocumentIntelligenceLoader,
    )
    from langchain_community.document_loaders.docugami import (
        DocugamiLoader,
    )
    from langchain_community.document_loaders.docusaurus import (
        DocusaurusLoader,
    )
    from langchain_community.document_loaders.dropbox import (
        DropboxLoader,
    )
    from langchain_community.document_loaders.duckdb_loader import (
        DuckDBLoader,
    )
    from langchain_community.document_loaders.email import (
        OutlookMessageLoader,
        UnstructuredEmailLoader,
    )
    from langchain_community.document_loaders.epub import (
        UnstructuredEPubLoader,
    )
    from langchain_community.document_loaders.etherscan import (
        EtherscanLoader,
    )
    from langchain_community.document_loaders.evernote import (
        EverNoteLoader,
    )
    from langchain_community.document_loaders.excel import (
        UnstructuredExcelLoader,
    )
    from langchain_community.document_loaders.facebook_chat import (
        FacebookChatLoader,
    )
    from langchain_community.document_loaders.fauna import (
        FaunaLoader,
    )
    from langchain_community.document_loaders.figma import (
        FigmaFileLoader,
    )
    from langchain_community.document_loaders.firecrawl import (
        FireCrawlLoader,
    )
    from langchain_community.document_loaders.gcs_directory import (
        GCSDirectoryLoader,
    )
    from langchain_community.document_loaders.gcs_file import (
        GCSFileLoader,
    )
    from langchain_community.document_loaders.geodataframe import (
        GeoDataFrameLoader,
    )
    from langchain_community.document_loaders.git import (
        GitLoader,
    )
    from langchain_community.document_loaders.gitbook import (
        GitbookLoader,
    )
    from langchain_community.document_loaders.github import (
        GithubFileLoader,
        GitHubIssuesLoader,
    )
    from langchain_community.document_loaders.glue_catalog import (
        GlueCatalogLoader,
    )
    from langchain_community.document_loaders.google_speech_to_text import (
        GoogleSpeechToTextLoader,
    )
    from langchain_community.document_loaders.googledrive import (
        GoogleDriveLoader,
    )
    from langchain_community.document_loaders.gutenberg import (
        GutenbergLoader,
    )
    from langchain_community.document_loaders.hn import (
        HNLoader,
    )
    from langchain_community.document_loaders.html import (
        UnstructuredHTMLLoader,
    )
    from langchain_community.document_loaders.html_bs import (
        BSHTMLLoader,
    )
    from langchain_community.document_loaders.hugging_face_dataset import (
        HuggingFaceDatasetLoader,
    )
    from langchain_community.document_loaders.hugging_face_model import (
        HuggingFaceModelLoader,
    )
    from langchain_community.document_loaders.ifixit import (
        IFixitLoader,
    )
    from langchain_community.document_loaders.image import (
        UnstructuredImageLoader,
    )
    from langchain_community.document_loaders.image_captions import (
        ImageCaptionLoader,
    )
    from langchain_community.document_loaders.imsdb import (
        IMSDbLoader,
    )
    from langchain_community.document_loaders.iugu import (
        IuguLoader,
    )
    from langchain_community.document_loaders.joplin import (
        JoplinLoader,
    )
    from langchain_community.document_loaders.json_loader import (
        JSONLoader,
    )
    from langchain_community.document_loaders.kinetica_loader import KineticaLoader
    from langchain_community.document_loaders.lakefs import (
        LakeFSLoader,
    )
    from langchain_community.document_loaders.larksuite import (
        LarkSuiteDocLoader,
    )
    from langchain_community.document_loaders.llmsherpa import (
        LLMSherpaFileLoader,
    )
    from langchain_community.document_loaders.markdown import (
        UnstructuredMarkdownLoader,
    )
    from langchain_community.document_loaders.mastodon import (
        MastodonTootsLoader,
    )
    from langchain_community.document_loaders.max_compute import (
        MaxComputeLoader,
    )
    from langchain_community.document_loaders.mediawikidump import (
        MWDumpLoader,
    )
    from langchain_community.document_loaders.merge import (
        MergedDataLoader,
    )
    from langchain_community.document_loaders.mhtml import (
        MHTMLLoader,
    )
    from langchain_community.document_loaders.modern_treasury import (
        ModernTreasuryLoader,
    )
    from langchain_community.document_loaders.mongodb import (
        MongodbLoader,
    )
    from langchain_community.document_loaders.news import (
        NewsURLLoader,
    )
    from langchain_community.document_loaders.notebook import (
        NotebookLoader,
    )
    from langchain_community.document_loaders.notion import (
        NotionDirectoryLoader,
    )
    from langchain_community.document_loaders.notiondb import (
        NotionDBLoader,
    )
    from langchain_community.document_loaders.obs_directory import (
        OBSDirectoryLoader,
    )
    from langchain_community.document_loaders.obs_file import (
        OBSFileLoader,
    )
    from langchain_community.document_loaders.obsidian import (
        ObsidianLoader,
    )
    from langchain_community.document_loaders.odt import (
        UnstructuredODTLoader,
    )
    from langchain_community.document_loaders.onedrive import (
        OneDriveLoader,
    )
    from langchain_community.document_loaders.onedrive_file import (
        OneDriveFileLoader,
    )
    from langchain_community.document_loaders.open_city_data import (
        OpenCityDataLoader,
    )
    from langchain_community.document_loaders.oracleadb_loader import (
        OracleAutonomousDatabaseLoader,
    )
    from langchain_community.document_loaders.oracleai import (
        OracleDocLoader,
        OracleTextSplitter,
    )
    from langchain_community.document_loaders.org_mode import (
        UnstructuredOrgModeLoader,
    )
    from langchain_community.document_loaders.pdf import (
        AmazonTextractPDFLoader,
        DedocPDFLoader,
        MathpixPDFLoader,
        OnlinePDFLoader,
        PagedPDFSplitter,
        PDFMinerLoader,
        PDFMinerPDFasHTMLLoader,
        PDFPlumberLoader,
        PyMuPDFLoader,
        PyPDFDirectoryLoader,
        PyPDFium2Loader,
        PyPDFLoader,
        UnstructuredPDFLoader,
    )
    from langchain_community.document_loaders.pebblo import (
        PebbloSafeLoader,
    )
    from langchain_community.document_loaders.polars_dataframe import (
        PolarsDataFrameLoader,
    )
    from langchain_community.document_loaders.powerpoint import (
        UnstructuredPowerPointLoader,
    )
    from langchain_community.document_loaders.psychic import (
        PsychicLoader,
    )
    from langchain_community.document_loaders.pubmed import (
        PubMedLoader,
    )
    from langchain_community.document_loaders.pyspark_dataframe import (
        PySparkDataFrameLoader,
    )
    from langchain_community.document_loaders.python import (
        PythonLoader,
    )
    from langchain_community.document_loaders.readthedocs import (
        ReadTheDocsLoader,
    )
    from langchain_community.document_loaders.recursive_url_loader import (
        RecursiveUrlLoader,
    )
    from langchain_community.document_loaders.reddit import (
        RedditPostsLoader,
    )
    from langchain_community.document_loaders.roam import (
        RoamLoader,
    )
    from langchain_community.document_loaders.rocksetdb import (
        RocksetLoader,
    )
    from langchain_community.document_loaders.rss import (
        RSSFeedLoader,
    )
    from langchain_community.document_loaders.rst import (
        UnstructuredRSTLoader,
    )
    from langchain_community.document_loaders.rtf import (
        UnstructuredRTFLoader,
    )
    from langchain_community.document_loaders.s3_directory import (
        S3DirectoryLoader,
    )
    from langchain_community.document_loaders.s3_file import (
        S3FileLoader,
    )
    from langchain_community.document_loaders.scrapfly import (
        ScrapflyLoader,
    )
    from langchain_community.document_loaders.sharepoint import (
        SharePointLoader,
    )
    from langchain_community.document_loaders.sitemap import (
        SitemapLoader,
    )
    from langchain_community.document_loaders.slack_directory import (
        SlackDirectoryLoader,
    )
    from langchain_community.document_loaders.snowflake_loader import (
        SnowflakeLoader,
    )
    from langchain_community.document_loaders.spider import (
        SpiderLoader,
    )
    from langchain_community.document_loaders.spreedly import (
        SpreedlyLoader,
    )
    from langchain_community.document_loaders.sql_database import (
        SQLDatabaseLoader,
    )
    from langchain_community.document_loaders.srt import (
        SRTLoader,
    )
    from langchain_community.document_loaders.stripe import (
        StripeLoader,
    )
    from langchain_community.document_loaders.surrealdb import (
        SurrealDBLoader,
    )
    from langchain_community.document_loaders.telegram import (
        TelegramChatApiLoader,
        TelegramChatFileLoader,
        TelegramChatLoader,
    )
    from langchain_community.document_loaders.tencent_cos_directory import (
        TencentCOSDirectoryLoader,
    )
    from langchain_community.document_loaders.tencent_cos_file import (
        TencentCOSFileLoader,
    )
    from langchain_community.document_loaders.tensorflow_datasets import (
        TensorflowDatasetLoader,
    )
    from langchain_community.document_loaders.text import (
        TextLoader,
    )
    from langchain_community.document_loaders.tidb import (
        TiDBLoader,
    )
    from langchain_community.document_loaders.tomarkdown import (
        ToMarkdownLoader,
    )
    from langchain_community.document_loaders.toml import (
        TomlLoader,
    )
    from langchain_community.document_loaders.trello import (
        TrelloLoader,
    )
    from langchain_community.document_loaders.tsv import (
        UnstructuredTSVLoader,
    )
    from langchain_community.document_loaders.twitter import (
        TwitterTweetLoader,
    )
    from langchain_community.document_loaders.unstructured import (
        UnstructuredAPIFileIOLoader,
        UnstructuredAPIFileLoader,
        UnstructuredFileIOLoader,
        UnstructuredFileLoader,
    )
    from langchain_community.document_loaders.url import (
        UnstructuredURLLoader,
    )
    from langchain_community.document_loaders.url_playwright import (
        PlaywrightURLLoader,
    )
    from langchain_community.document_loaders.url_selenium import (
        SeleniumURLLoader,
    )
    from langchain_community.document_loaders.vsdx import (
        VsdxLoader,
    )
    from langchain_community.document_loaders.weather import (
        WeatherDataLoader,
    )
    from langchain_community.document_loaders.web_base import (
        WebBaseLoader,
    )
    from langchain_community.document_loaders.whatsapp_chat import (
        WhatsAppChatLoader,
    )
    from langchain_community.document_loaders.wikipedia import (
        WikipediaLoader,
    )
    from langchain_community.document_loaders.word_document import (
        Docx2txtLoader,
        UnstructuredWordDocumentLoader,
    )
    from langchain_community.document_loaders.xml import (
        UnstructuredXMLLoader,
    )
    from langchain_community.document_loaders.xorbits import (
        XorbitsLoader,
    )
    from langchain_community.document_loaders.youtube import (
        GoogleApiClient,
        GoogleApiYoutubeLoader,
        YoutubeLoader,
    )
    from langchain_community.document_loaders.yuque import (
        YuqueLoader,
    )


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
    "BrowserbaseLoader": "langchain_community.document_loaders.browserbase",
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
    "DedocAPIFileLoader": "langchain_community.document_loaders.dedoc",
    "DedocFileLoader": "langchain_community.document_loaders.dedoc",
    "DedocPDFLoader": "langchain_community.document_loaders.pdf",
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
    "KineticaLoader": "langchain_community.document_loaders.kinetica_loader",
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
    "OracleDocLoader": "langchain_community.document_loaders.oracleai",
    "OracleTextSplitter": "langchain_community.document_loaders.oracleai",
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
    "ScrapflyLoader": "langchain_community.document_loaders.scrapfly",
    "SQLDatabaseLoader": "langchain_community.document_loaders.sql_database",
    "SRTLoader": "langchain_community.document_loaders.srt",
    "SeleniumURLLoader": "langchain_community.document_loaders.url_selenium",
    "SharePointLoader": "langchain_community.document_loaders.sharepoint",
    "SitemapLoader": "langchain_community.document_loaders.sitemap",
    "SlackDirectoryLoader": "langchain_community.document_loaders.slack_directory",
    "SnowflakeLoader": "langchain_community.document_loaders.snowflake_loader",
    "SpiderLoader": "langchain_community.document_loaders.spider",
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
    "BrowserbaseLoader",
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
    "DedocAPIFileLoader",
    "DedocFileLoader",
    "DedocPDFLoader",
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
    "GlueCatalogLoader",
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
    "ImageCaptionLoader",
    "IMSDbLoader",
    "IuguLoader",
    "JoplinLoader",
    "JSONLoader",
    "KineticaLoader",
    "LakeFSLoader",
    "LarkSuiteDocLoader",
    "LLMSherpaFileLoader",
    "MastodonTootsLoader",
    "MHTMLLoader",
    "MWDumpLoader",
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
    "OracleDocLoader",
    "OracleTextSplitter",
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
    "ScrapflyLoader",
    "SQLDatabaseLoader",
    "SRTLoader",
    "SeleniumURLLoader",
    "SharePointLoader",
    "SitemapLoader",
    "SlackDirectoryLoader",
    "SnowflakeLoader",
    "SpiderLoader",
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
